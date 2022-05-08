import random

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, args):
        self._copy_args(args)
        self._load_files()
        self._print_info()
        self._build_dicts()
        self._precalculate_normalization()

    def _copy_args(self, args):
        self.path = args.data
        self.slurm = args.slurm
        self.logger = args.logger
        self.device = args.device
        self.batch_size = args.batch_size
        self.neg_samples = args.neg_samples

    def _load_files(self):
        self.logger.info('loading data')
        self.train_df = pd.read_table(self.path + 'train.tsv', header=0, names=['user_id', 'asin'])
        self.test_df = pd.read_table(self.path + 'test.tsv', header=0, names=['user_id', 'asin'])
        self.item_mapping = pd.read_csv(self.path + 'item_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.user_mapping = pd.read_csv(self.path + 'user_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.train_df.user_id = self.train_df.user_id.map(dict(self.user_mapping.values))
        self.test_df.user_id = self.test_df.user_id.map(dict(self.user_mapping.values))
        self.train_df.asin = self.train_df.asin.map(dict(self.item_mapping.values))
        self.test_df.asin = self.test_df.asin.map(dict(self.item_mapping.values))

    def _build_dicts(self):
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].aggregate(list).to_dict()
        self.test_user_dict = self.test_df.groupby('user_id')['asin'].aggregate(list).to_dict()
        self.positive_lists = [self.train_user_dict[u] for u in range(self.n_users)]

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
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo().astype(np.float)
        self.neighbors = [torch.tensor(i) for i in norm_adj.tolil().rows]
        self.norm_matrix = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)

    def _adjacency_matrix(self):
        ''' create bipartite graph with initial vectors '''
        graph = dgl.heterograph({('item', 'bought_by', 'user'): (self.train_df['asin'].values,
                                                                 self.train_df['user_id'].values),
                                 ('user', 'bought', 'item'): (self.train_df['user_id'].values,
                                                              self.train_df['asin'].values)})
        user_ids = torch.tensor(list(range(self.n_users)), dtype=torch.long)
        item_ids = torch.tensor(self.item_mapping.index, dtype=torch.long)
        graph.ndata['id'] = {'user': user_ids,
                             'item': item_ids}
        return graph.adj(etype='bought', scipy_fmt='coo', ctx=self.device)

    def _convert_sp_mat_to_sp_tensor(self, coo):
        ''' convert sparse matrix into torch sparse tensor '''
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _print_info(self):
        self.n_users = self.train_df.user_id.nunique()
        self.n_items = self.train_df.asin.nunique()
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.n_batches = (self.n_train - 1) // self.batch_size + 1
        self.bucket_len = self.n_train // self.n_users
        self.iterable_len = self.bucket_len * self.n_users

        self.logger.info(f"n_train:    {self.n_train:-7}")
        self.logger.info(f"n_test:     {self.test_df.shape[0]:-7}")
        self.logger.info(f"n_users:    {self.n_users:-7}")
        self.logger.info(f"n_items:    {self.n_items:-7}")
        self.logger.info(f"n_batches:  {self.n_batches:-7}")

    def get_user_pos_items(self, users):
        ''' returns positive items per batch of users '''
        return [self.train_user_dict[user] for user in users]

    ''' this is done for compatibility w torch Dataset class '''

    def __len__(self):
        return self.iterable_len

    def __getitem__(self, idx):
        '''
            each user has a continuous 'bucket'
            user_id depends on the bucket number
        '''
        idx //= self.bucket_len
        pos = random.choice(self.positive_lists[idx])
        pos_set = set(self.positive_lists[idx])

        if len(pos_set) + self.neg_samples > self.n_items:
            self.logger.warn(f"user {idx} doesn't have enough items for negative sampling")

        neg_set = set()
        while len(neg_set) < self.neg_samples:
            neg = random.choice(range(self.n_items))
            if neg not in pos_set:
                neg_set.add(neg)
        return torch.tensor([idx, pos] + list(neg_set)).to(self.device)
