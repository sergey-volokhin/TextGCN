import os
import random
import sys

import dgl
import numpy as np
import pandas as pd
import torch
from cppimport import imp_from_filepath

from utils import minibatch, shuffle


class MyDataLoader:

    def __init__(self, args):
        self._copy_args(args)
        self._load_files()
        self._print_info()
        self._build_dicts()
        self._construct_graph()

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

    def _construct_graph(self):
        ''' create bipartite graph with initial vectors '''
        self.graph = dgl.heterograph({('item', 'bought_by', 'user'): (self.train_df['asin'].values,
                                                                      self.train_df['user_id'].values),
                                      ('user', 'bought', 'item'): (self.train_df['user_id'].values,
                                                                   self.train_df['asin'].values)})
        self.graph = self.graph.to(self.device)
        user_ids = torch.tensor(list(range(self.n_users)), dtype=torch.long).to(self.device)
        item_ids = torch.tensor(self.item_mapping['remap_id'], dtype=torch.long).to(self.device)
        self.graph.ndata['id'] = {'user': user_ids,
                                  'item': item_ids}

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
        self.logger.info(f"n_test:     {self.test_df.shape[0]:-7}")
        self.logger.info(f"n_batches:  {self.n_batches:-7}")
        # self.logger.info(f"sparsity:   {(1.0 - (self.n_train * 1.0))/(self.n_users * self.n_items)*100}")
        # self.logger.info(f'sparsity:   {(self.n_train + self.n_test)/(self.n_users * self.n_items)}')

    def sampler(self, neg_ratio=1):
        '''
            sample n_train users with their positive and negative items.
            S is a (n_train, 3)-shaped np.array of triples (user, pos, neg),
            yields 3 batched lists: [user1, ...], [pos1, ...], [neg1, ...]
        '''
        S = self.c_sampler.sample_negative(self.n_users,
                                           self.n_items,
                                           self.n_train,
                                           self.positive_lists,
                                           neg_ratio)

        users = torch.Tensor(S[:, 0]).to(self.device).long()
        pos_items = torch.Tensor(S[:, 1]).to(self.device).long()
        neg_items = torch.Tensor(S[:, 2]).to(self.device).long()
        return minibatch(*shuffle(users, pos_items, neg_items), batch_size=self.batch_size)

    def get_user_pos_items(self, users):
        ''' returns positive items per batch of users '''
        return [self.train_user_dict[user] for user in users]

    # def __len__(self):
    #     return self.n_train

    # def __getitem__(self, idx):
    #     pos = random.choice(self.positive_lists[idx])
    #     pos_set = set(self.positive_lists[idx])
    #     while True:
    #         neg = random.choice(range(self.n_items))
    #         if neg not in pos_set:
    #             return idx, pos, neg

    # def __iter__(self):
    #     pass
