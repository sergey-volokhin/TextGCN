import os
import random
import sys

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from cppimport import imp_from_filepath
from torch.utils.data import Dataset

from utils import minibatch, shuffle


class MyDataLoader(Dataset):

    def __init__(self, args):
        self._copy_args(args)
        self._load_files()
        self._print_info()
        self._build_dicts()
        self._precalculate_normalization()
        # self._calculate_centralities()

    def _calculate_centralities(self):
        import networkx as nx
        self.train_df['user_id'] += self.train_df['asin'].values.max() + 1
        nx_g = nx.Graph(self.train_df[['asin', 'user_id']].values.tolist())

        for name, function in [
            ('density', nx.density),
            ('average_clustering', nx.average_clustering),
            ('degree_centrality', nx.degree_centrality),
            ('eigenvector_centrality', nx.eigenvector_centrality),
            # ('katz_centrality', nx.katz_centrality),                  #   2 hours
            # ('closeness_centrality', nx.closeness_centrality),        #  29 hours for amazon-book
            # ('harmonic_centrality', nx.harmonic_centrality),          #  37 hours
            # ('load_centrality', nx.load_centrality),                  # 126 hours
            # ('betweenness_centrality', nx.betweenness_centrality),    # 197 hours
            # ('information_centrality', nx.information_centrality),                                  # coo tocsc
            # ('communicability_betweenness_centrality', nx.communicability_betweenness_centrality),  # sp.linalg.expm
            # ('second_order_centrality', nx.second_order_centrality),                                # np.linalg.solve
            # ('percolation_centrality', nx.percolation_centrality),                                  # keyerror 0
        ]:
            try:
                if name == 'eigenvector_centrality':
                    res = function(nx_g, max_iter=1000)
                else:
                    res = function(nx_g)
                if isinstance(res, float):
                    self.logger.info(f'{name:25} {res}')
                elif isinstance(res, dict):
                    array = np.fromiter(res.values(), dtype=float)
                    self.logger.info(f'{name:25} mean: {array.mean():>10.6f}, med: {np.median(array):10.6f}')
            except Exception:
                self.logger.info(f'no {name} :c')
                raise
        self.train_df['user_id'] -= self.train_df['asin'].values.max() + 1
        exit()

    def _copy_args(self, args):
        self.seed = args.seed
        self.path = args.data
        self.logger = args.logger
        self.device = args.device
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
        random.seed(self.seed)

    def _build_dicts(self):
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].aggregate(list).to_dict()
        self.test_user_dict = self.test_df.groupby('user_id')['asin'].aggregate(list).to_dict()
        self.positive_lists = [self.train_user_dict[u] for u in range(self.n_users)]

    def _adjacency_matrix(self):
        ''' create bipartite graph with initial vectors '''
        graph = dgl.heterograph({('item', 'bought_by', 'user'): (self.train_df['asin'].values,
                                                                 self.train_df['user_id'].values),
                                 ('user', 'bought', 'item'): (self.train_df['user_id'].values,
                                                              self.train_df['asin'].values)})
        # graph = graph.to(self.device)
        user_ids = torch.tensor(list(range(self.n_users)), dtype=torch.long)  # .to(self.device)
        item_ids = torch.tensor(self.item_mapping['remap_id'], dtype=torch.long)  # .to(self.device)
        graph.ndata['id'] = {'user': user_ids,
                             'item': item_ids}
        return graph.adj(etype='bought', scipy_fmt='coo', ctx=self.device)

    def _convert_sp_mat_to_sp_tensor(self, x):
        ''' convert sparse matrix into torch sparse tensor '''
        coo = x.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _precalculate_normalization(self):
        '''
            precalculate normalization coefficients:
                            1
            c_(ij) = ----------------
                     sqrt(|N_u||N_i|)
        '''
        adj_mat = self._adjacency_matrix()
        adj_mat._shape = (self.n_users + self.n_items, self.n_users + self.n_items)
        adj_mat.col += self.n_users
        adj_mat = (adj_mat + adj_mat.T).todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
        self.norm_matrix = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce()

    def _print_info(self):
        self.n_users = self.train_df.user_id.nunique()
        self.n_items = self.train_df.asin.nunique()
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.n_batches = (self.n_train - 1) // self.batch_size + 1

        self.logger.info(f"n_train:    {self.n_train:-7}")
        self.logger.info(f"n_test:     {self.test_df.shape[0]:-7}")
        self.logger.info(f"n_users:    {self.n_users:-7}")
        self.logger.info(f"n_items:    {self.n_items:-7}")
        self.logger.info(f"n_batches:  {self.n_batches:-7}")

    def sampler(self, neg_ratio=1):
        '''
            sample n_train users with their positive and negative items.
            each users gets n_train//n_users pairs of (pos, neg) samples
            S is an np.array of triples (user, pos, neg),
            yields 3 batched lists: [user1, ...], [pos1, ...], [neg1, ...]
        '''
        S = self.c_sampler.sample_negative(self.n_users,
                                           self.n_items,
                                           self.n_train,
                                           self.positive_lists,
                                           neg_ratio)

        users = torch.Tensor(S[:, 0]).long()
        pos_items = torch.Tensor(S[:, 1]).long()
        neg_items = torch.Tensor(S[:, 2]).long()

        # users = torch.Tensor(S[:, 0]).to(self.device).long()
        # pos_items = torch.Tensor(S[:, 1]).to(self.device).long()
        # neg_items = torch.Tensor(S[:, 2]).to(self.device).long()
        return minibatch(*shuffle(users, pos_items, neg_items), batch_size=self.batch_size)

    def get_user_pos_items(self, users):
        ''' returns positive items per batch of users '''
        return [self.train_user_dict[user] for user in users]

    ''' this is done for compatibility w torch Dataset class '''

    def __len__(self):
        return self.n_train // self.n_users * self.n_users

    def __getitem__(self, idx):
        idx //= (self.n_train // self.n_users)
        pos = random.choice(self.positive_lists[idx])
        pos_set = set(self.positive_lists[idx])
        while True:
            neg = random.choice(range(self.n_items))
            if neg not in pos_set:
                return torch.tensor([idx, pos, neg])
