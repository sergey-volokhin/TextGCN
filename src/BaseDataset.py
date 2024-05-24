import os
import random
from collections import defaultdict, deque
from itertools import repeat
from os.path import join

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, config):
        super().__init__()
        self._copy_params(config)
        self._load_files(config.reshuffle)
        self._convert_to_internal_ids()
        self._build_dicts()
        self._print_info()
        self._precalculate_normalization()
        self._safety_checks()

    def _safety_checks(self):
        items_not_in_train = set(self.test_df['asin'].unique()) - set(self.train_df['asin'].unique())
        assert not items_not_in_train, f"test set contains items that are not in the train set: {items_not_in_train}"
        users_not_in_train = set(self.test_df['user_id'].unique()) - set(self.train_df['user_id'].unique())
        assert not users_not_in_train, f"test set contains users that are not in the train set: {users_not_in_train}"

        if hasattr(self, 'val_df'):
            items_not_in_train = set(self.val_df['asin'].unique()) - set(self.train_df['asin'].unique())
            assert not items_not_in_train, f"test has items that are not in the train set: {items_not_in_train}"
            users_not_in_train = set(self.val_df['user_id'].unique()) - set(self.train_df['user_id'].unique())
            assert not users_not_in_train, f"test has users that are not in the train set: {users_not_in_train}"

    def _copy_params(self, config):
        self.path: str = config.data
        self.slurm: bool = config.slurm
        self.device: torch.device = config.device
        self.batch_size: int = config.batch_size
        self.neg_samples: int = config.neg_samples
        self.seed: int = config.seed
        self.logger = config.logger

    def _load_files(self, reshuffle: bool):
        ''' if valid doesn't exist, use test as valid set and not have test set '''
        self.logger.debug('loading data')

        folder = self.path
        if reshuffle or not os.path.exists(join(folder, 'train.tsv')):
            folder = join(self.path, f'reshuffle_{self.seed}')
            if not os.path.exists(join(folder, 'train.tsv')):
                return self._reshuffle_train_test()

        self.train_df = (
            pd.read_table(join(folder, 'train.tsv'), dtype=str)
            .sort_values(by=['user_id', 'asin'])
            .reset_index(drop=True)
        )
        self.test_df = (
            pd.read_table(join(folder, 'test.tsv'), dtype=str)
            .sort_values(by=['user_id', 'asin'])
            .reset_index(drop=True)
        )
        if os.path.exists(join(folder, 'valid.tsv')):
            self.logger.debug('loading validation set')
            self.val_df = (
                pd.read_table(join(folder, 'valid.tsv'), dtype=str)
                .sort_values(by=['user_id', 'asin'])
                .reset_index(drop=True)
            )

    def _read_data_to_reshuffle(self):
        '''
        read and return full data for reshuffling
        files train, test, and valid if they exist, reviews_text.tsv otherwise
        filter to have at least 3 items per user
        '''
        if os.path.exists(join(self.path, 'reviews_text.tsv')):
            df = pd.read_table(join(self.path, 'reviews_text.tsv'), dtype=str).dropna()[['user_id', 'asin', 'rating', 'time']]
        else:
            train_df = pd.read_table(join(self.path, 'train.tsv'), dtype=str)
            test_df = pd.read_table(join(self.path, 'test.tsv'), dtype=str)
            df = pd.concat([train_df, test_df])
            if os.path.exists(join(self.path, 'valid.tsv')):
                val_df = pd.read_table(join(self.path, 'valid.tsv'), dtype=str)
                df = pd.concat([df, val_df])

        vc = df['user_id'].value_counts()
        return df[df['user_id'].isin(vc[vc > 1].index)]

    def _train_test_split(self, df):
        ''' split df into train-test or train-val-test depending on objective '''
        raise NotImplementedError

    def _reshuffle_train_test(self):
        self.logger.debug('reshuffling train-test')
        os.makedirs(join(self.path, f'reshuffle_{self.seed}'), exist_ok=True)

        df = self._read_data_to_reshuffle()
        self._train_test_split(df)

        self.train_df = self.train_df.sort_values(by=['user_id', 'asin']).reset_index(drop=True)
        self.train_df.to_csv(join(self.path, f'reshuffle_{self.seed}/train.tsv'), sep='\t', index=False)

        self.test_df = self.test_df.sort_values(by=['user_id', 'asin']).reset_index(drop=True)
        self.test_df = self.test_df[self.test_df['asin'].isin(self.train_df['asin'].unique())]
        self.test_df.to_csv(join(self.path, f'reshuffle_{self.seed}/test.tsv'), sep='\t', index=False)

        if hasattr(self, 'val_df'):
            self.val_df = self.val_df.sort_values(by=['user_id', 'asin']).reset_index(drop=True)
            self.val_df = self.val_df[self.val_df['asin'].isin(self.train_df['asin'].unique())]
            self.val_df.to_csv(join(self.path, f'reshuffle_{self.seed}/valid.tsv'), sep='\t', index=False)

    def _convert_to_internal_ids(self):
        self.user_mapping = pd.DataFrame(enumerate(self.train_df.user_id.unique()), columns=['remap_id', 'org_id'])
        self.item_mapping = pd.DataFrame(enumerate(self.train_df.asin.unique()), columns=['remap_id', 'org_id'])
        self.user_mapping = self.user_mapping.astype({'remap_id': int, 'org_id': str})
        self.item_mapping = self.item_mapping.astype({'remap_id': int, 'org_id': str})

        self.train_df.user_id = self.train_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.test_df.user_id = self.test_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.train_df.asin = self.train_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))
        self.test_df.asin = self.test_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))

        if hasattr(self, 'val_df'):
            self.val_df.user_id = self.val_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
            self.val_df.asin = self.val_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))

    def _build_dicts(self):
        ''' build dicts for fast lookup '''
        self.n_users = self.train_df.user_id.nunique()
        self.n_items = self.train_df.asin.nunique()
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        if hasattr(self, 'val_df'):
            self.n_val = self.val_df.shape[0]
        self.bucket_len = self.n_train // self.n_users  # number of samples per user
        self.iterable_len = self.bucket_len * self.n_users  # length of torch Dataset we convert into
        self.all_items = range(self.n_items)  # this needs to be a python list for correct set conversion

        self.cached_samplings = defaultdict(deque)

        self.train_user_dict = self.train_df.groupby('user_id')['asin'].aggregate(list)
        self.positive_lists = [{
            'list': self.train_user_dict[u],  # list for faster random.choice
            'set': set(self.train_user_dict[u]),  # set for faster "x in y" check
            'tensor': torch.tensor(self.train_user_dict[u]).to(self.device),  # tensor for faster set difference
        } for u in range(self.n_users)]

        self.user_representations = {}  # todo: where should this be?
        self.item_representations = {}  # todo: where should this be?

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
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo().astype(np.float64)
        self.norm_matrix = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce()

    def _adjacency_matrix(self):
        ''' create bipartite graph with initial vectors '''
        graph = dgl.heterograph({
            ('item', 'bought_by', 'user'): (self.train_df['asin'].values, self.train_df['user_id'].values),
            ('user', 'bought', 'item'): (self.train_df['user_id'].values, self.train_df['asin'].values),
        }, device=self.device)
        self.user_ids = torch.tensor(range(self.n_users), dtype=torch.long, device=self.device)
        self.item_ids = torch.tensor(range(self.n_items), dtype=torch.long, device=self.device)
        graph.ndata['id'] = {'user': self.user_ids, 'item': self.item_ids}
        return graph.adj_external(etype='bought', scipy_fmt='coo', ctx=self.device)

    def _convert_sp_mat_to_sp_tensor(self, coo):
        ''' convert sparse matrix into torch sparse tensor '''
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), device=self.device)

    def _print_info(self):
        self.logger.info(f"n_train:    {self.n_train:-7}")
        if hasattr(self, 'n_val'):
            self.logger.info(f"n_val:      {self.n_val:-7}")
        self.logger.info(f"n_test:     {self.n_test:-7}")
        self.logger.info(f"n_users:    {self.n_users:-7}")
        self.logger.info(f"n_items:    {self.n_items:-7}")

    def _cache_samples(self, idx: int):  # todo: add structured_negative_sampling from torch_geom
        '''
        precaching pos and neg samples for users to save time during iteration
        sample exactly as many examples at the beginning of each epoch as we'll need per item
        '''
        positives = random.choices(self.positive_lists[idx]['list'], k=self.bucket_len)
        neg_samples = set()
        while len(neg_samples) < self.bucket_len * self.neg_samples:
            neg_sample = random.choice(self.all_items)
            if neg_sample not in self.positive_lists[idx]['set']:
                neg_samples.add(neg_sample)
        negatives = np.array(list(neg_samples)).reshape(-1, self.bucket_len)
        self.cached_samplings[idx] = deque(torch.tensor(list(zip(repeat(idx), positives, *negatives))))

    def __len__(self):
        return self.iterable_len

    def __getitem__(self, idx: int):
        '''
        to iterate over entire train_df per epoch, total length of the dataset should be n_train, not n_users
        hence each user has a continuous 'bucket' of self.n_train // self.n_users items,
        incoming idx is the id of the element. to find the id of the bucket (i.e. user), divide by its length
        '''
        idx //= self.bucket_len
        if not self.cached_samplings[idx]:
            self._cache_samples(idx)
        return self.cached_samplings[idx].pop()
