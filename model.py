import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from base_model import BaseModel
from metrics import l2_loss_mean
from utils import minibatch


class ConvLayer(nn.Module):

    def __init__(self, in_feats, out_feats, mess_dropout, activation=nn.ReLU(), nonlinearity='relu'):
        super(ConvLayer, self).__init__()
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

    def forward(self, graph, feat, user_vector=None, get_attention=False):
        with graph.local_scope():  # entering local mode to not accidentally change anything when propagating through layer

            if user_vector is None:
                h_src = feat[graph.ndata['id']['user']]
            else:
                h_src = user_vector.expand(graph.num_nodes('user'), feat.shape[1])

            h_dst = feat[graph.ndata['id']['item']]

            ''' compute attention aka get alpha (on edges) '''
            graph.ndata['e_u/i'] = {'user': h_src, 'item': h_dst}
            graph['bought'].apply_edges(fn.u_dot_v('e_u/i', 'e_u/i', 'alpha'))
            graph['bought'].edata['alpha'] /= self._in_feats ** (1 / 2)
            soft_alpha = torch.softmax(graph.edata.pop('alpha')[('user', 'bought', 'item')], dim=0)
            graph['bought_by'].edata['alpha'] = graph['bought'].edata['alpha'] = soft_alpha

            ''' get user and item vectors updated '''
            # get layer based on the type of vertex
            feat_src = self.fc_src(h_src).view(-1, self._out_feats)  # W_1 * e_u + b_1
            feat_dst = self.fc_dst(h_dst).view(-1, self._out_feats)  # W_2 * e_j + b_2
            feat_src = self.activation(feat_src)  # g(W_1 * e_u + b_1)
            feat_dst = self.activation(feat_dst)  # g(W_2 * e_j + b_2)
            graph.ndata['g(We+b)'] = {'user': feat_src, 'item': feat_dst}

            graph['bought'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))  # sum att over item-neighbors
            graph['bought_by'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))  # sum att over user-neighbors

            rst = self.mess_drop(torch.cat([graph.ndata['e_new']['user'], graph.ndata['e_new']['item']]))
            if get_attention:
                return rst, graph.edata['alpha']
            return rst


class Model(BaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        self._copy_args(args)
        self._copy_dataset_args(dataset)
        self._build_layers()
        self.load_model()

        self.to(self.device)
        self._build_optimizer()
        self.logger.info(self)

    def _copy_args(self, args):
        super()._copy_args(args)
        self.reg_lambda = args.reg_lambda
        self.layer_size = args.layer_size
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.n_batches = dataset.num_batches
        self.user_vector = dataset.user_vector
        self.train_user_dict = dataset.train_user_dict
        self.entity_embeddings = dataset.entity_embeddings
        self.sampler = dataset.sampler

    def _build_layers(self):
        ''' aggregation layers '''
        self.layers = nn.ModuleList()
        for k in range(len(self.layer_size) - 1):
            self.layers.append(ConvLayer(self.layer_size[k], self.layer_size[k + 1], self.mess_dropout))

    def gnn(self):
        ''' recalculate embeddings '''
        g = self.graph.local_var()  # entering local mode to not accidentally change anything while calculating embeddings
        h = self.entity_embeddings.weight[:]
        node_embed_cache = [h]  # we need to remove the first layer representations of the user nodes

        ''' first layer processed using user_vector '''
        h = self.layers[0](g, h, self.user_vector)
        out = F.normalize(h, p=2, dim=1)
        node_embed_cache.append(out)

        ''' rest of the layers processed using user embeddings '''
        for layer in self.layers[1:]:
            h = layer(g, h)
            out = F.normalize(h, p=2, dim=1)
            node_embed_cache.append(out)
        return torch.cat(node_embed_cache, 1)

    def get_loss(self, users, pos, neg):
        embedding = self.gnn()
        src_vec = embedding[users]
        pos_dst_vec = embedding[pos]
        neg_dst_vec = embedding[neg]
        pos_score = torch.bmm(src_vec.unsqueeze(1), pos_dst_vec.unsqueeze(2)).squeeze()
        neg_score = torch.bmm(src_vec.unsqueeze(1), neg_dst_vec.unsqueeze(2)).squeeze()
        loss = torch.mean(F.logsigmoid(pos_score - neg_score)) * (-1.0)
        reg_loss = l2_loss_mean(src_vec) + l2_loss_mean(pos_dst_vec) + l2_loss_mean(neg_dst_vec)
        return loss + self.reg_lambda * reg_loss

    def get_sampler(self):
        return self.sampler(), self.n_batches

    def predict(self, save=False):
        ''' this should be a model-agnostic method '''

        users = list(self.test_user_dict)
        y_pred, y_true = [], []
        with torch.no_grad():  # don't calculate gradient since we only predict
            items_emb, users_emb = torch.split(self.gnn(), [self.n_items, self.n_users])
            for batch_users in tqdm(minibatch(users, batch_size=self.batch_size),
                                    total=len(users) // self.batch_size + 1,
                                    desc='test batches',
                                    leave=False,
                                    dynamic_ncols=True):

                batch_user_emb = users_emb[torch.Tensor(batch_users).long().to(self.device)]
                rating = torch.matmul(batch_user_emb, items_emb.t())
                exclude_index, exclude_items = [], []
                pos_items = [self.train_user_dict[u] for u in batch_users]
                for ind, items in enumerate(pos_items):
                    exclude_index += [ind] * len(items)
                    exclude_items.append(items)
                exclude_items = np.concatenate(exclude_items)
                rating[exclude_index, exclude_items] = np.NINF
                _, rank_indices = torch.topk(rating, k=max(self.k))
                y_pred += list(rank_indices.cpu().numpy().tolist())
                y_true += [self.test_user_dict[u] for u in batch_users]

        predictions = pd.DataFrame.from_dict({'y_pred': y_pred, 'y_true': y_true})
        if save:
            predictions.to_csv(f'{self.save_path}/predictions.tsv', sep='\t', index=False)
            self.logger.info(f'Predictions are saved in `{self.save_path}/predictions.tsv`')
        return predictions
