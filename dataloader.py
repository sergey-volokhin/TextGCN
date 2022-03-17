import dgl
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from cppimport import imp_from_filepath
import scipy.sparse as sp

from utils import minibatch, shuffle


class DataLoader:

    def __init__(self, args):
        self._copy_args(args)
        self._load_files()
        self._print_info()
        self._build_dicts()
        self._init_embeddings()
        self._construct_graph()
        self._precalculate_normalization()

    def _copy_args(self, args):
        self.seed = args.seed
        self.path = args.data
        self.logger = args.logger
        self.device = args.device
        self.emb_size = args.emb_size
        self.keep_prob = args.keep_prob
        self.batch_size = args.batch_size

    def _load_files(self):
        self.logger.info('loading data')
        self.train_df = pd.read_table(self.path + 'train.tsv', dtype=np.int32, header=0, names=['user_id', 'asin'])
        self.test_df = pd.read_table(self.path + 'test.tsv', dtype=np.int32, header=0, names=['user_id', 'asin'])
        self.item_mapping = pd.read_csv(self.path + 'item_list.txt', sep=' ')[['org_id', 'remap_id']]

        ''' load C++ sampling '''
        sys.path.append('sources')
        self.c_sampler = imp_from_filepath(os.path.join(os.path.dirname(__file__), "sources/sampling.cpp"))
        self.c_sampler.seed(self.seed)

    def _init_embeddings(self):
        ''' randomly initialize entity embeddings '''
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.n_users,
                                                 embedding_dim=self.emb_size).to(self.device)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.n_items,
                                                 embedding_dim=self.emb_size).to(self.device)
        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)

    def _build_dicts(self):
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].aggregate(list).to_dict()
        self.test_user_dict = self.test_df.groupby('user_id')['asin'].aggregate(list).to_dict()
        self.positive_lists = [self.train_user_dict[u] for u in range(self.n_users)]

    def _print_info(self):
        self.n_users = self.train_df.user_id.nunique()
        self.n_items = self.train_df.asin.nunique()
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.n_batches = (self.n_train - 1) // self.batch_size + 1

        self.logger.info(f"n_users:    {self.n_users:-7}")
        self.logger.info(f"n_items:    {self.n_items:-7}")
        self.logger.info(f"n_train:    {self.n_train:-7}")
        self.logger.info(f"n_test:     {self.n_test:-7}")
        self.logger.info(f"n_batches:  {self.n_batches:-7}")

    def _construct_graph(self):
        ''' create bipartite graph with initial vectors '''
        self.graph = dgl.heterograph({('item', 'bought_by', 'user'): (self.train_df['asin'].values,
                                                                      self.train_df['user_id'].values),
                                      ('user', 'bought', 'item'): (self.train_df['user_id'].values,
                                                                   self.train_df['asin'].values)})
        self.graph = self.graph.to(self.device)
        user_ids = torch.tensor(list(range(self.n_users)), dtype=torch.long).to(self.device)
        item_ids = torch.tensor(self.item_mapping['remap_id'], dtype=torch.long).to(self.device)
        self.graph.ndata['e'] = {'user': self.embedding_user(user_ids),
                                 'item': self.embedding_item(item_ids)}
        self.graph.ndata['id'] = {'user': user_ids,
                                  'item': item_ids}

    def _precalculate_normalization(self):
        '''
            precalculate normalization coefficients:
                            1
            c_(ij) = ----------------
                     sqrt(|N_u||N_i|)
        '''
        adj_mat = self.graph.adj(etype='bought', scipy_fmt='coo', ctx=self.device)
        adj_mat._shape = (self.n_users + self.n_items, self.n_users + self.n_items)
        adj_mat.col += self.n_users
        adj_mat = (adj_mat + adj_mat.T).todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
        self.norm_matrix = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.norm_matrix = self.norm_matrix.coalesce()

    def _convert_sp_mat_to_sp_tensor(self, x):
        ''' convert sparse matrix into torch sparse tensor '''
        coo = x.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _dropout_norm_matrix(self):
        ''' drop (1 - self.keep_prob) elements from adjacency table '''
        index = self.norm_matrix.indices().t()
        values = self.norm_matrix.values()
        random_index = (torch.rand(len(values)) + self.keep_prob).int().bool()
        index = index[random_index]
        values = values[random_index] / self.keep_prob
        return torch.sparse.FloatTensor(index.t(), values, self.norm_matrix.size()).to(self.device)

    def sampler(self, neg_ratio=1):
        '''
            sample n_train users with their positive and negative items
            S: a (n_train, 3) shaped np.array of triples (user, pos, neg)
            yields 3 batched lists: [user1, ...], [pos1, ...], [neg1, ...]
        '''
        try:
            S = self.c_sampler.sample_negative(self.n_users,
                                               self.n_items,
                                               self.n_train,
                                               self.positive_lists,
                                               neg_ratio)
        except Exception:
            S = self.python_sampler()

        users = torch.Tensor(S[:, 0]).to(self.device).long()
        pos_items = torch.Tensor(S[:, 1]).to(self.device).long()
        neg_items = torch.Tensor(S[:, 2]).to(self.device).long()
        return minibatch(*shuffle(users, pos_items, neg_items), batch_size=self.batch_size)

    def python_sampler(self):
        ''' fallback python sampling: 100x slower than C++ '''
        self.logger.warn('C++ sampling not available, falling back to slow python')
        users = np.random.randint(0, self.n_users, self.n_train)
        result = []
        for user in users:
            pos_for_user = self.positive_lists[user]
            pos = np.random.choice(pos_for_user, 1)
            while True:
                neg = random.randint(0, self.n_items)
                if neg not in pos_for_user:
                    break
            result.append([user, pos, neg])
        return np.array(result)

    def get_user_pos_items(self, users):
        ''' returns positive items per batch of users '''
        return [self.train_user_dict[user] for user in users]
