import os
from collections import defaultdict

import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch.softmax import edge_softmax
from tqdm import tqdm

from metric import calculate_metrics, l2_loss_mean


class MarcusGATConv(nn.Module):

    def __init__(self, in_feats, out_feats, mess_dropout, num_heads=1, activation=nn.ReLU()):
        super(MarcusGATConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        ''' src = users
            dst = items '''
        self.fc_src = nn.Linear(in_feats, out_feats, bias=True)
        self.fc_dst = nn.Linear(in_feats, out_feats, bias=True)

        self._out_feats = out_feats
        self._num_heads = num_heads
        self.activation = activation
        self.mess_drop = nn.Dropout(mess_dropout)

        # initialize weights
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.constant_(self.fc_src.bias, 0)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.constant_(self.fc_dst.bias, 0)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if isinstance(feat, tuple):
                h_src = feat[0]
                h_dst = feat[1]
            else:
                h_src = h_dst = feat

            ''' compute attention aka get alpha (on edges) '''
            graph.srcdata.update({'e_u': h_src})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'e_i': h_dst})
            graph.apply_edges(fn.u_mul_v('e_u', 'e_i', 'e'))
            e = graph.edata.pop('e')
            graph.edata['alpha'] = edge_softmax(graph, e)  # TODO: divide by sqrt(d_k)

            ''' get user and item vectors updated '''
            # get layer based on the type of vertex
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)  # W_1 * e_u + b_1
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)  # W_2 * e_j + b_2
            feat_src = self.activation(feat_src)  # g(W_1 * e_u + b_1)
            feat_dst = self.activation(feat_dst)  # g(W_2 * e_j + b_2)
            graph.srcdata.update({'g(We+b)': feat_src})
            graph.dstdata.update({'g(We+b)': feat_dst})

            graph.update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))  # sum over item-neighbors
            # graph.update_all(fn.e_mul_v('alpha', 'g(We+b)', 'm'), fn.sum('m', 'e_new'))  # sum over item-neighbors?
            # graph.update_all(fn.e_mul_u('alpha', 'g(We+b)', 'm'), fn.sum('m', 'e_new'))  # sum over user-neighbors?

            rst = self.mess_drop(graph.dstdata['e_new'])
            if get_attention:
                return rst, graph.edata['alpha']
            return rst


class Model(nn.Module):

    def __init__(self, args, dataset):
        super(Model, self).__init__()
        self.args = args
        self.logger = args.logger

        self._copy_args(args)
        self._copy_dataset_args(dataset)
        self._build_layers(args)
        self._build_graph()
        # self._build_embeddings(args)
        self.load_model(args)
        self.metrics_logger = defaultdict(lambda: np.zeros((0, len(args.k))))

        self.to(self.device)
        self._build_optimizer(args)
        self.logger.info(self)

    def _build_optimizer(self, args):
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, patience=5, min_lr=1e-7)

    def _copy_args(self, args):
        self.device = args.device
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.k = args.k
        self.evaluate_every = args.evaluate_every
        self.reg_lambda_kg = args.regs
        self.reg_lambda_gnn = args.regs
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.sampler_mode = args.sampler_mode
        self.uid = args.uid

    def _copy_dataset_args(self, dataset):
        self.dataset = dataset
        self.logger = dataset.logger
        self.n_entities = dataset.n_entities
        self.n_relations = dataset.n_kg_relation
        self.item_id_range = torch.LongTensor(dataset.item_id_range).to(self.device)
        self.train_user_dict = dataset.train_user_dict
        self.test_user_dict = dataset.test_user_dict
        self.train_g = dataset.train_g
        self.num_kg_batches = dataset.n_train_kg_triplet // self.kg_batch_size + 1
        if self.sampler_mode == 'unique':
            self.num_cf_batches = dataset.n_train
        else:
            self.num_cf_batches = dataset.cf_graph.number_of_edges()  # num_samples comes from dgl.edgeSampler
        self.num_cf_batches //= self.cf_batch_size + 1

    def _build_layers(self, args):
        ''' aggregation layers '''
        self.layers = nn.ModuleList()
        for k in range(len(args.layer_size) - 1):
            self.layers.append(MarcusGATConv(args.layer_size[k], args.layer_size[k + 1], args.mess_dropout[k]))

    def _build_graph(self):
        self.train_g.ndata['id'] = torch.LongTensor(self.train_g.ndata['id']).to(self.device)
        self.train_g.edata['type'] = torch.LongTensor(self.train_g.edata['type']).to(self.device)
        # self.train_g.edata['w'] = torch.LongTensor(self.train_g.edata['w']).to(self.device)

    def gnn(self):
        ''' recalculate embeddings '''
        g = self.train_g.local_var()
        h = self.entity_embed(g.ndata['id'])
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
        cf_loss = torch.mean(F.logsigmoid(pos_score - neg_score)) * (-1.0)
        reg_loss = l2_loss_mean(src_vec) + l2_loss_mean(pos_dst_vec) + l2_loss_mean(neg_dst_vec)
        return cf_loss + self.reg_lambda_gnn * reg_loss

    # def _build_embeddings(self, args):
    #     self.entity_embed = nn.Embedding(self.n_entities, args.entity_embed_dim)  # e_h, e_t
    #     self.relation_embed = nn.Embedding(self.n_relations, args.relation_embed_dim)  # e_r
    #     self.W_Rs = nn.Parameter(torch.Tensor(self.n_relations, args.entity_embed_dim, args.relation_embed_dim))  # w_r
    #     nn.init.xavier_uniform_(self.W_Rs, gain=nn.init.calculate_gain('leaky_relu', 0.2))

    # def _att_score(self, edges):
    #     ''' att_score = (w_r h_t)^T tanh(w_r h_r + e_r) '''
    #     t_r = torch.matmul(self.entity_embed(edges.src['id']), self.w_r)
    #     h_r = torch.matmul(self.entity_embed(edges.dst['id']), self.w_r)
    #     att_w = torch.bmm(t_r.unsqueeze(1), torch.tanh(h_r + self.relation_embed(edges.data['type'])).unsqueeze(2))
    #     return {'att_w': att_w.squeeze(-1)}

    # # def compute_attention(self, g):
    #     ''' compute attention weight and store it on edges '''
    #     g = g.local_var()
    #     for i in range(self.n_relations):
    #         e_idxs = g.filter_edges(lambda edges: edges.data['type'] == i)
    #         self.w_r = self.W_Rs[i]
    #         g.apply_edges(self._att_score, e_idxs)
    #     return edge_softmax(g, g.edata.pop('att_w'))

    # def evaluate(self):
    #     self.eval()
    #     with torch.no_grad():
    #         self.train_g.edata['w'] = self.compute_attention(self.train_g)
    #         ret = calculate_metrics(self.gnn(),  # embeddings
    #                                 self.train_user_dict,
    #                                 self.test_user_dict,
    #                                 self.item_id_range,
    #                                 self.k)
    #     self.logger.info('            ' + ''.join([f'@{i:<6}' for i in self.k]))
    #     for i in ret:
    #         self.metrics_logger[i] = np.append(self.metrics_logger[i], [ret[i]], axis=0)
    #         self.logger.info(f"{i} {' '*(9-len(i))} " + ' '.join([f'{j:.4f}' for j in ret[i]]))
    #     self.save_progression()
    #     return ret

    def save_progression(self):
        ''' save all scores in a file '''
        epochs_string = '           '
        for i in range(self.evaluate_every, self.epochs + 1, self.evaluate_every):
            epochs_string += f'%-{7 * len(self.k) + 1}s' % f'{i} epochs'
        at_string = '           ' + ((''.join([f'@{i:<6}' for i in self.k]) + ' ') * len(self.metrics_logger['recall']))
        progression = ['Model ' + str(self.uid), 'Full progression:', epochs_string, at_string]
        for k, v in self.metrics_logger.items():
            progression.append(f'{k:10} ' + '  '.join([' '.join([f'{g:.4f}' for g in j]) for j in v]))
        progression = '\n'.join(progression)
        open(self.progression_path, 'w').write(progression)

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
                score = torch.matmul(embedding[u_id], embedding[self.item_id_range].transpose(0, 1))
                score[self.train_user_dict[u_id]] = 0.0
                _, rank_indices = torch.topk(score, k=max(self.k), largest=True)
                rank_indices = rank_indices.cpu().numpy()
                result.append([rank_indices.tolist(), pos_item_l.tolist()])
        prediction = pd.DataFrame(result, columns=['predicted', 'true_test'])
        prediction.to_csv(os.path.join(self.save_path, 'predictions.tsv'), sep='\t', index=False)
