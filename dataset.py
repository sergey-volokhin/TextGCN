import os
import random
from collections import defaultdict, deque
from itertools import repeat

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset):

    def __init__(self, args):
        self._copy_args(args)
        self._load_files(args.reshuffle, args.seed)
        self._print_info()
        self._convert_to_internal_ids()
        self._build_dicts()
        self._precalculate_normalization()

    def _copy_args(self, args):
        self.old = args.old
        self.path = args.data
        self.slurm = args.slurm
        self.logger = args.logger
        self.device = args.device
        self.batch_size = args.batch_size
        self.neg_samples = args.neg_samples

    def _load_files(self, reshuffle, seed):  # todo simplify loading
        self.logger.info('loading data')

        if reshuffle and os.path.isdir(self.path + f'reshuffle_{seed}'):
            path = self.path + f'reshuffle_{seed}/'
            self.train_df = pd.read_table(path + 'train.tsv', header=0, names=['user_id', 'asin'])
            self.test_df = pd.read_table(path + 'test.tsv', header=0, names=['user_id', 'asin'])
        else:
            self.train_df = pd.read_table(self.path + 'train.tsv', header=0, names=['user_id', 'asin'])
            self.test_df = pd.read_table(self.path + 'test.tsv', header=0, names=['user_id', 'asin'])
            if reshuffle:
                self._reshuffle_train_test(seed)

        assert not(set(self.test_df['asin'].unique()) -
                   set(self.train_df['asin'].unique())), "item from test set doesn't appear in train set"

        assert not(set(self.test_df['user_id'].unique()) -
                   set(self.train_df['user_id'].unique())), "user from test set doesn't appear in train set"

    def _reshuffle_train_test(self, seed):
        os.makedirs(self.path + f'reshuffle_{seed}', exist_ok=True)

        train, test = [], []
        for _, group in tqdm(pd.concat([self.train_df, self.test_df]).groupby('user_id'),
                             desc='reshuffling',
                             dynamic_ncols=True,
                             leave=False,
                             disable=self.slurm):
            if len(group) < 3:
                continue
            s_train, s_test = tts(group, test_size=0.2, random_state=seed)
            train.append(s_train)
            test.append(s_test)

        self.train_df = pd.concat(train)
        self.test_df = pd.concat(test)
        self.train_df.to_csv(self.path + f'reshuffle_{seed}/train.tsv', sep='\t', index=False)
        self.test_df.to_csv(self.path + f'reshuffle_{seed}/test.tsv', sep='\t', index=False)

    def _convert_to_internal_ids(self):

        if self.old:
            self.user_mapping = pd.read_csv(self.path + 'user_list.txt', sep=' ')[['org_id', 'remap_id']]
            self.item_mapping = pd.read_csv(self.path + 'item_list.txt', sep=' ')[['org_id', 'remap_id']]
        else:
            self.user_mapping = pd.DataFrame(enumerate(self.train_df.user_id.unique()), columns=['remap_id', 'org_id'])
            self.item_mapping = pd.DataFrame(enumerate(self.train_df.asin.unique()), columns=['remap_id', 'org_id'])

        self.train_df.user_id = self.train_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.test_df.user_id = self.test_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.train_df.asin = self.train_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))
        self.test_df.asin = self.test_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))

    def _build_dicts(self):
        self.cached_samplings = defaultdict(list)
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].aggregate(list)
        self.positive_lists = [{
            'list': self.train_user_dict[u],                                 # list for faster random.choice
            'set': set(self.train_user_dict[u]),                             # set for faster "x in y" check
            'tensor': torch.tensor(self.train_user_dict[u]).to(self.device)  # tensor for faster set difference
        } for u in range(self.n_users)]

        # split test into batches once at init instead of at every predict
        test_user_agg = self.test_df.groupby('user_id')['asin'].aggregate(list)
        users = test_user_agg.index.to_list()
        self.test_batches = [users[j:j + self.batch_size] for j in range(0, len(users), self.batch_size)]

        # list of lists with test samples (per user)
        self.true_test_lil = test_user_agg.values.tolist()

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
        self.norm_matrix = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)

    def _adjacency_matrix(self):
        ''' create bipartite graph with initial vectors '''
        graph = dgl.heterograph({('item', 'bought_by', 'user'): (self.train_df['asin'].values,
                                                                 self.train_df['user_id'].values),
                                 ('user', 'bought', 'item'): (self.train_df['user_id'].values,
                                                              self.train_df['asin'].values)})
        user_ids = torch.tensor(list(range(self.n_users)), dtype=torch.long)
        item_ids = torch.tensor(range(self.n_items), dtype=torch.long)
        graph.ndata['id'] = {'user': user_ids, 'item': item_ids}
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
        self.bucket_len = self.n_train // self.n_users      # number of samples per user
        self.iterable_len = self.bucket_len * self.n_users  # length of torch Dataset we convert into
        self.all_items = range(self.n_items)  # this needs to be a python list for correct set conversion

        self.logger.info(f"n_train:    {self.n_train:-7}")
        self.logger.info(f"n_test:     {self.test_df.shape[0]:-7}")
        self.logger.info(f"n_users:    {self.n_users:-7}")
        self.logger.info(f"n_items:    {self.n_items:-7}")

    ''' this is done for compatibility w torch Dataset class '''

    def __len__(self):
        return self.iterable_len

    def __getitem__(self, idx):
        ''' each user has a continuous 'bucket', user_id depends on the bucket number '''
        idx //= self.bucket_len

        '''
            precaching pos and neg samples for users to save time during iteration
            sample exactly as many examples at the beginning of each epoch as we'll need per item
        '''
        if not self.cached_samplings[idx]:
            positives = random.choices(self.positive_lists[idx]['list'], k=self.bucket_len)
            while True:
                negatives = random.choices(self.all_items, k=self.bucket_len * self.neg_samples)
                if len(set(negatives).intersection(self.positive_lists[idx]['set'])) == 0:
                    break
            negatives = [negatives[i:i + self.bucket_len] for i in range(0, len(negatives), self.bucket_len)]
            self.cached_samplings[idx] = deque(torch.tensor(list(zip(repeat(idx), positives, *negatives))))

        return self.cached_samplings[idx].pop()
