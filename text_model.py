# ! borked

import os

import dgl.function as fn
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from base_model import BaseModel
from dataset import BaseDataset
from utils import embed_text


class DatasetKG(BaseDataset):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.sep = args.sep
        self.freeze = args.freeze
        self.bert_model = args.bert_model
        self.emb_batch_size = args.emb_batch_size

    def _load_files(self):
        super()._load_files()
        self.kg_df_text = pd.read_table(self.path + 'kg_readable.tsv', dtype=str)[['asin', 'relation', 'attribute']]

    def _construct_text_representation(self):
        self.item_text_dict = {}
        for asin, group in tqdm(self.kg_df_text.groupby('asin'), desc='construct text repr', dynamic_ncols=True):
            vals = group[['relation', 'attribute']].values
            self.item_text_dict[asin] = f' {self.sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping['text'] = self.item_mapping['org_id'].map(self.item_text_dict)

    def _init_embeddings(self):
        ''' construct BERT representation for items and overwrite model item embeddings '''
        super()._init_embeddings()

        emb_path = f'embeddings_{self.bert_model.split("/")[-1]}.torch'
        if not os.path.exists(self.path + emb_path):
            self._construct_text_representation()
            embeddings = embed_text(self.item_mapping['text'],
                                    'item_kg_representation',
                                    self.path,
                                    self.bert_model,
                                    self.emb_batch_size,
                                    self.device,
                                    self.logger)
        else:
            embeddings = torch.load(self.path + emb_path, map_location=self.device)

        embeddings = torch.tensor(embeddings).to(self.device)
        self.embedding_item = torch.nn.Embedding.from_pretrained(embeddings, freeze=self.freeze)


class ConvLayer(nn.Module):

    def __init__(self, in_dim, out_dim, keep_prob, activation=nn.ReLU(), nonlinearity='relu'):
        super(ConvLayer, self).__init__()
        '''
            src = users
            dst = items
        '''
        self.fc_src = nn.Linear(in_dim, out_dim, bias=True)
        self.fc_dst = nn.Linear(in_dim, out_dim, bias=True)

        self._in_dim = in_dim
        self._out_dim = out_dim
        self.activation = activation
        self.dropout = nn.Dropout(keep_prob)

        ''' initialize weights'''
        # non-linearity types: sigmoid, tanh, relu, leaky_relu, selu''
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
            graph['bought'].edata['alpha'] /= self._in_dim ** (1 / 2)
            soft_alpha = torch.softmax(graph.edata.pop('alpha')[('user', 'bought', 'item')], dim=0)
            graph['bought_by'].edata['alpha'] = graph['bought'].edata['alpha'] = soft_alpha

            ''' get user and item vectors updated '''
            # get layer based on the type of vertex
            feat_src = self.activation(self.fc_src(h_src).view(-1, self._out_dim))  # g(W_1 * e_u + b_1)
            feat_dst = self.activation(self.fc_dst(h_dst).view(-1, self._out_dim))  # g(W_1 * e_u + b_1)
            graph.ndata['g(We+b)'] = {'user': feat_src, 'item': feat_dst}

            # sum att over item-neighbors
            graph['bought'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))
            # sum att over user-neighbors
            graph['bought_by'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))

            rst = self.dropout(torch.cat([graph.ndata['e_new']['user'], graph.ndata['e_new']['item']]))
            if get_attention:
                return rst, graph.edata['alpha']
            return rst


class TextModelKG(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.layer_sizes = [args.emb_size] + args.layer_sizes
        self.single_vector = args.single_vector

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.train_user_dict = dataset.train_user_dict

    def _add_torch_vars(self):

        ''' aggregation layers '''
        self.layers = nn.ModuleList()
        for k in range(len(self.layer_sizes) - 1):
            self.layers.append(ConvLayer(self.layer_sizes[k], self.layer_sizes[k + 1], self.keep_prob))

        ''' set up the initial user vector if only having one vector for all users '''
        if self.single_vector:
            self.user_vector = torch.nn.parameter.Parameter(torch.Tensor(self.emb_size))
            torch.nn.init.xavier_uniform_(self.user_vector.unsqueeze(1))
        else:
            self.user_vector = None

    @property
    def representation(self):
        g = self.graph.local_var()  # entering local mode to freeze the graph for computations

        h = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        # norm_matrix = self._dropout_norm_matrix
        node_embed_cache = [h]

        ''' first layer processed using user_vector '''
        h = self.layers[0](g, h, self.user_vector)  # add norm_matrix multiplication
        node_embed_cache.append(h)

        ''' rest of the layers processed using user embeddings '''
        for layer in self.layers[1:]:
            h = layer(g, h)
            # h = self.layer_propagate(norm_matrix, h)
            node_embed_cache.append(h)
        aggregated_embeddings = self.layer_aggregation(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def layer_aggregation(self, vectors):
        return torch.cat(vectors, 1)
