# ! borked

import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from base_model import BaseModel
from dataset import BaseDataset
from utils import embed_text


class DatasetKG(BaseDataset):

    def __init__(self, args):
        super().__init__(args)
        self._construct_text_representation()

    def _copy_args(self, args):
        super()._copy_args(args)
        self.sep = args.sep

    def _load_files(self):
        super()._load_files()
        self.kg_df_text = pd.read_table(self.path + 'kg_readable.tsv', dtype=str)[['asin', 'relation', 'attribute']]

    def _construct_text_representation(self):
        item_text_dict = {}
        for asin, group in tqdm(self.kg_df_text.groupby('asin'), desc='construct text repr', dynamic_ncols=True):
            vals = group[['relation', 'attribute']].values
            item_text_dict[asin] = f' {self.sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping['text'] = self.item_mapping['org_id'].map(item_text_dict)


class TextModelKG(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.freeze = args.freeze
        self.bert_model = args.bert_model
        self.single_vector = args.single_vector
        self.emb_batch_size = args.emb_batch_size
        self.layer_sizes = [args.emb_size] + args.layer_sizes

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.item_mapping = dataset.item_mapping
        self.train_user_dict = dataset.train_user_dict

    def _add_vars(self):
        # linear layer for formula
        gain = nn.init.calculate_gain('relu')
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True, device=self.device)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.constant_(self.linear.bias, 0.0)

    def _init_embeddings(self, emb_size):
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

        # ''' aggregation layers '''
        # self.layers = nn.ModuleList()
        # for k in range(len(self.layer_sizes) - 1):
            # self.layers.append(ConvLayer(self.layer_sizes[k], self.layer_sizes[k + 1], self.keep_prob))

        # ''' set up the initial user vector if only having one vector for all users '''
        # if self.single_vector:
        #     self.user_vector = torch.nn.parameter.Parameter(torch.Tensor(self.emb_size))
        #     torch.nn.init.xavier_uniform_(self.user_vector.unsqueeze(1))
        # else:
        #     self.user_vector = None

    @property
    def representation(self):

        curent_lvl_emb_matrix = self.embedding_matrix
        norm_matrix = self._dropout_norm_matrix
        node_embed_cache = [curent_lvl_emb_matrix]

        for _ in range(self.n_layers):
            curent_lvl_emb_matrix = self.layer_propagate(norm_matrix, curent_lvl_emb_matrix)
            node_embed_cache.append(curent_lvl_emb_matrix)

        # g = self.graph.local_var()  # entering local mode to freeze the graph for computations
        # curent_lvl_emb_matrix = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        # node_embed_cache = [curent_lvl_emb_matrix]

        # ''' first layer processed using user_vector '''
        # h = self.layers[0](g, h, self.user_vector)  # add norm_matrix multiplication
        # node_embed_cache.append(h)

        # ''' rest of the layers processed using user embeddings '''
        # for layer in self.layers[1:]:
        #     h = layer(g, curent_lvl_emb_matrix)
        #     # h = self.layer_propagate(norm_matrix, h)
        #     node_embed_cache.append(curent_lvl_emb_matrix)

        aggregated_embeddings = self.layer_aggregation(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def layer_aggregation(self, vectors):
        return torch.cat(vectors, 1)

    def layer_propagate(self, norm_matrix, emb_matrix):
        return torch.sparse.mm(norm_matrix, emb_matrix)

    #     alpha = F.softmax(torch.sparse.mm(norm_matrix, emb_matrix) / torch.sqrt(self.emb_size))
    #     self.g(
    #         self.linear()
    #     )

    # def g(self):
    #     pass



# class ConvLayer(nn.Module):

#     def __init__(self, in_dim, out_dim, keep_prob, activation=nn.ReLU(), nonlinearity='relu'):
#         super(ConvLayer, self).__init__()
#         '''
#             src = users
#             dst = items
#         '''
#         self.fc_src = nn.Linear(in_dim, out_dim, bias=True)
#         self.fc_dst = nn.Linear(in_dim, out_dim, bias=True)

#         self._in_dim = in_dim
#         self._out_dim = out_dim
#         self.activation = activation
#         self.dropout = nn.Dropout(keep_prob)

#         ''' initialize weights'''
#         # non-linearity types: sigmoid, tanh, relu, leaky_relu, selu''
#         gain = nn.init.calculate_gain(nonlinearity)
#         nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
#         nn.init.constant_(self.fc_src.bias, 0)
#         nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
#         nn.init.constant_(self.fc_dst.bias, 0)

#     def forward(self, graph, feat, user_vector=None, get_attention=False):
#         # entering local mode to not accidentally change anything when propagating through layer
#         with graph.local_scope():

#             if user_vector is None:
#                 h_src = feat[graph.ndata['id']['user']]
#             else:
#                 h_src = user_vector.expand(graph.num_nodes('user'), feat.shape[1])

#             h_dst = feat[graph.ndata['id']['item']]

#             ''' compute attention aka get alpha (on edges) '''
#             graph.ndata['e_u/i'] = {'user': h_src, 'item': h_dst}
#             graph['bought'].apply_edges(fn.u_dot_v('e_u/i', 'e_u/i', 'alpha'))
#             graph['bought'].edata['alpha'] /= self._in_dim ** (1 / 2)
#             soft_alpha = torch.softmax(graph.edata.pop('alpha')[('user', 'bought', 'item')], dim=0)
#             graph['bought_by'].edata['alpha'] = graph['bought'].edata['alpha'] = soft_alpha

#             ''' get user and item vectors updated '''
#             # get layer based on the type of vertex
#             feat_src = self.activation(self.fc_src(h_src).view(-1, self._out_dim))  # g(W_1 * e_u + b_1)
#             feat_dst = self.activation(self.fc_dst(h_dst).view(-1, self._out_dim))  # g(W_1 * e_u + b_1)
#             graph.ndata['g(We+b)'] = {'user': feat_src, 'item': feat_dst}

#             # sum att over item-neighbors
#             graph['bought'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))
#             # sum att over user-neighbors
#             graph['bought_by'].update_all(fn.u_mul_e('g(We+b)', 'alpha', 'm'), fn.sum('m', 'e_new'))

#             rst = self.dropout(torch.cat([graph.ndata['e_new']['user'], graph.ndata['e_new']['item']]))
#             if get_attention:
#                 return rst, graph.edata['alpha']
#             return rst
