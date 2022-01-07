import os
from collections import defaultdict

import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from metric import calculate_metrics, l2_loss_mean


class MarcusGATConv(nn.Module):

    def __init__(self, in_feats, out_feats, mess_dropout, activation=nn.ReLU(), nonlinearity='relu'):
        super(MarcusGATConv, self).__init__()
        '''
            src = users
            dst = items
        '''
        self.fc_src = nn.Linear(in_feats, out_feats, bias=True)
        self.fc_dst = nn.Linear(in_feats, out_feats, bias=True)

        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.mess_drop = nn.Dropout(mess_dropout)

        # initialize weights
        ''' non-linearity types: sigmoid, tanh, relu, leaky_relu, selu'''
        gain = nn.init.calculate_gain(nonlinearity)
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.constant_(self.fc_src.bias, 0)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.constant_(self.fc_dst.bias, 0)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():  # entering local mode to not accidentally change anything when propagating through layer

            h_src = feat[graph.ndata['id']['user']]
            h_dst = feat[graph.ndata['id']['item']]

            ''' compute attention aka get alpha (on edges) '''
            graph.ndata['e_u/i'] = {'user': h_src, 'item': h_dst}
            graph['bought'].apply_edges(fn.u_dot_v('e_u/i', 'e_u/i', 'alpha'))
            graph['bought'].edata['alpha'] /= self._in_feats ** (1 / 2)
            alpha = graph.edata.pop('alpha')
            graph['bought_by'].edata['alpha'] = graph['bought'].edata['alpha'] = torch.softmax(alpha[('user', 'bought', 'item')], dim=0)

            ''' get user and item vectors updated '''
            # get layer based on the type of vertex
            feat_src = self.fc_src(h_src).view(-1, self._out_feats)  # W_1 * e_u + b_1
            feat_dst = self.fc_dst(h_dst).view(-1, self._out_feats)  # W_2 * e_j + b_2
            feat_src = self.activation(feat_src)  # g(W_1 * e_u + b_1)
            feat_dst = self.activation(feat_dst)  # g(W_2 * e_j + b_2)
            graph.ndata['g(We+b)'] = {'user': feat_src, 'item': feat_dst}

            graph['bought'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))  # sum over item-neighbors
            graph['bought_by'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))  # sum over user-neighbors

            rst = self.mess_drop(torch.cat([graph.ndata['e_new']['user'], graph.ndata['e_new']['item']]))
            if get_attention:
                return rst, graph.edata['alpha']
            return rst


class Model(nn.Module):

    def __init__(self, args, dataset):
        super(Model, self).__init__()
        self.logger = args.logger

        self._copy_args(args)
        self._copy_dataset_args(dataset)
        self._build_layers(args)
        self.load_model(args)
        self.metrics_logger = defaultdict(lambda: np.zeros((0, len(args.k))))

        self.to(self.device)
        self._build_optimizer(args)
        # self.logger.info(self)

    def _copy_args(self, args):
        self.device = args.device
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.k = args.k
        self.evaluate_every = args.evaluate_every
        self.reg_lambda = args.reg_lambda
        self.uid = args.uid

    def _copy_dataset_args(self, dataset):
        self.logger = dataset.logger
        self.dataset = dataset
        self.graph = dataset.graph
        self.n_entities = dataset.n_entities
        self.entity_embeddings = dataset.entity_embeddings
        self.train_user_dict = dataset.train_user_dict
        self.test_user_dict = dataset.test_user_dict

    def _build_layers(self, args):
        ''' aggregation layers '''
        self.layers = nn.ModuleList()
        for k in range(len(args.layer_size) - 1):
            self.layers.append(MarcusGATConv(args.layer_size[k], args.layer_size[k + 1], args.mess_dropout))

    def _build_optimizer(self, args):
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=(not args.quiet), patience=5, min_lr=1e-6)

    def gnn(self):
        ''' recalculate embeddings '''
        g = self.graph.local_var()  # entering local mode to not accidentally change anything while calculating embeddings
        h = self.entity_embeddings.weight[:]
        node_embed_cache = [h]
        for layer in self.layers:
            h = layer(g, h)
            out = F.normalize(h, p=2, dim=1)
            node_embed_cache.append(out)
        return torch.cat(node_embed_cache, 1)

    def get_loss(self, src_ids, pos_dst_ids, neg_dst_ids):
        embedding = self.gnn()
        src_vec = embedding[src_ids]
        pos_dst_vec = embedding[pos_dst_ids]
        neg_dst_vec = embedding[neg_dst_ids]
        pos_score = torch.bmm(src_vec.unsqueeze(1), pos_dst_vec.unsqueeze(2)).squeeze()  # (batch_size, )
        neg_score = torch.bmm(src_vec.unsqueeze(1), neg_dst_vec.unsqueeze(2)).squeeze()  # (batch_size, )
        loss = torch.mean(F.logsigmoid(pos_score - neg_score)) * (-1.0)
        reg_loss = l2_loss_mean(src_vec) + l2_loss_mean(pos_dst_vec) + l2_loss_mean(neg_dst_vec)
        return loss + self.reg_lambda * reg_loss

    def evaluate(self):
        self.eval()
        with torch.no_grad():
            ret = calculate_metrics(self.gnn(),  # embeddings
                                    self.train_user_dict,
                                    self.test_user_dict,
                                    self.graph.ndata['id']['item'],
                                    self.k)
        self.logger.info('            ' + ''.join([f'@{i:<6}' for i in self.k]))
        for i in ret:
            self.metrics_logger[i] = np.append(self.metrics_logger[i], [ret[i]], axis=0)
            self.logger.info(f'{i:11}' + ' '.join([f'{j:.4f}' for j in ret[i]]))
        self.save_progression()
        return ret

    def save_progression(self):
        ''' save all scores in a file '''
        epochs_string, at_string = [' ' * 9], [' ' * 9]
        width = max(10, len(self.k) * 7 - 1)
        for i in range(len(self.metrics_logger['recall'])):
            epochs_string.append(f'%-{width}s' % f'{(i + 1) * self.evaluate_every} epochs')
            at_string.append(f'%-{width}s' % ' '.join([f'@{i:<5}' for i in self.k]))
        progression = ['Model ' + str(self.uid), 'Full progression:', '  '.join(epochs_string), '  '.join(at_string)]
        for k, v in self.metrics_logger.items():
            progression.append(f'{k:11}' + '  '.join([f'%-{width}s' % ' '.join([f'{g:.4f}' for g in j]) for j in v]))
        open(self.progression_path, 'w').write('\n'.join(progression))

    def checkpoint(self, epoch):
        ''' save current model and update best '''
        os.system(f'rm -f {self.save_path}model_checkpoint*')
        torch.save(self.state_dict(), self.save_path + f'model_checkpoint{epoch}')
        if max(self.metrics_logger['recall'][:, 0]) == self.metrics_logger['recall'][-1][0]:
            self.logger.info(f'Updating best model at epoch {epoch}')
            torch.save(self.state_dict(), self.save_path + 'model_best')

    def load_model(self, args):
        if args.load_model:
            self.load_state_dict(torch.load(args.load_model))
            self.logger.info('Loaded model ' + args.load_model)
        index = max([0] + [int(i[12:-4]) for i in os.listdir(self.save_path) if i.startswith('progression_')])
        self.progression_path = os.path.join(self.save_path, f'progression_{index + 1}.txt')

    def predict(self):
        result = []
        embedding = self.gnn()
        with torch.no_grad():
            for u_id, pos_item_l in tqdm(self.test_user_dict.items(), dynamic_ncols=True, leave=False, desc='predicting'):
                score = torch.matmul(embedding[u_id], embedding[self.graph.ndata['id']['item']].transpose(0, 1))
                score[self.train_user_dict[u_id]] = 0.0
                _, rank_indices = torch.topk(score, k=max(self.k), largest=True)
                rank_indices = rank_indices.cpu().numpy()
                result.append([rank_indices.tolist(), pos_item_l.tolist()])
        prediction = pd.DataFrame(result, columns=['predicted', 'true_test'])
        prediction.to_csv(os.path.join(self.save_path, 'predictions.tsv'), sep='\t', index=False)
        self.logger.info('Predictions are saved in ' + os.path.join(self.save_path, 'predictions.tsv'))
