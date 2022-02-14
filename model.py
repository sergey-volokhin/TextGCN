import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseModel


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
        # entering local mode to not accidentally change anything when propagating through layer
        with graph.local_scope():

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
            feat_src = self.activation(self.fc_src(h_src).view(-1, self._out_feats))  # g(W_1 * e_u + b_1)
            feat_dst = self.activation(self.fc_dst(h_dst).view(-1, self._out_feats))  # g(W_1 * e_u + b_1)
            graph.ndata['g(We+b)'] = {'user': feat_src, 'item': feat_dst}

            graph['bought'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))  # sum att over item-neighbors
            graph['bought_by'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))  # sum att over user-neighbors

            rst = self.mess_drop(torch.cat([graph.ndata['e_new']['user'], graph.ndata['e_new']['item']]))
            if get_attention:
                return rst, graph.edata['alpha']
            return rst


class ModelText(BaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

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

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.user_vector = dataset.user_vector
        self.train_user_dict = dataset.train_user_dict
        self.embedding_user = dataset.embedding_user
        self.embedding_item = dataset.embedding_item
        self.get_user_pos_items = dataset.get_user_pos_items

    def _build_layers(self):
        ''' aggregation layers '''
        self.layers = nn.ModuleList()
        for k in range(len(self.layer_size) - 1):
            self.layers.append(ConvLayer(self.layer_size[k], self.layer_size[k + 1], self.mess_dropout))

    def get_representation(self):
        # entering local mode to not accidentally change anything while calculating embeddings
        g = self.graph.local_var()
        h = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
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
        node_embed_cache = torch.cat(node_embed_cache, 1)

        return torch.split(node_embed_cache, [self.n_users, self.n_items])
