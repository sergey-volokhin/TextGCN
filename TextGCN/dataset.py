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


class BaseDataset(Dataset):

    def __init__(self, params):
        self._copy_params(params)
        self._load_files(params.reshuffle)
        self._convert_to_internal_ids()
        self._build_dicts()
        self._print_info()
        self._precalculate_normalization()

        assert self.n_items > max(params.k), f'all k must be less than number of items ({self.n_items}), got k={params.k}'

    def _copy_params(self, params):
        self.path: str = params.data
        self.slurm: bool = params.slurm
        self.device = params.device
        self.batch_size: int = params.batch_size
        self.neg_samples: int = params.neg_samples
        self.seed: int = params.seed
        self.logger = params.logger

    def _load_files(self, reshuffle: bool):
        self.logger.info('loading data')

        folder = self.path
        if reshuffle:
            folder = os.path.join(self.path, f'reshuffle_{self.seed}')
            if not os.path.exists(os.path.join(folder, 'train.tsv')):
                return self._reshuffle_train_test()

        self.train_df = (
            pd.read_table(os.path.join(folder, 'train.tsv'), dtype=str)
            .sort_values(by=['user_id', 'asin'])
            .reset_index(drop=True)
        )
        self.test_df = (
            pd.read_table(os.path.join(folder, 'test.tsv'), dtype=str)
            .sort_values(by=['user_id', 'asin'])
            .reset_index(drop=True)
        )

        items_only_in_test = set(self.test_df['asin'].unique()) - set(self.train_df['asin'].unique())
        assert not items_only_in_test, f"items {items_only_in_test} from test set don't appear in train set"
        users_only_in_test = set(self.test_df['user_id'].unique()) - set(self.train_df['user_id'].unique())
        assert not users_only_in_test, f"users {users_only_in_test} from test set doesn't appear in train set"

    def _reshuffle_train_test(self, train_size: float = 0.8):
        self.logger.info('reshuffling train-test')
        os.makedirs(os.path.join(self.path, f'reshuffle_{self.seed}'), exist_ok=True)

        train_df = pd.read_table(os.path.join(self.path, 'train.tsv'), dtype=str)
        test_df = pd.read_table(os.path.join(self.path, 'test.tsv'), dtype=str)

        df = pd.concat([train_df, test_df])
        group_sizes = df.groupby('user_id').size()
        valid_groups = group_sizes[group_sizes >= 3].index
        filtered_df = df[df['user_id'].isin(valid_groups)]
        self.train_df, self.test_df = tts(
            filtered_df,
            stratify=filtered_df['user_id'],
            train_size=train_size,
            random_state=self.seed,
        )
        self.train_df = self.train_df.sort_values(by=['user_id', 'asin']).reset_index(drop=True)
        self.test_df = self.test_df.sort_values(by=['user_id', 'asin']).reset_index(drop=True)

        ''' remove items from test that don't appear in train '''
        self.test_df = self.test_df[self.test_df['asin'].isin(self.train_df['asin'].unique())]

        self.train_df.to_csv(os.path.join(self.path, f'reshuffle_{self.seed}/train.tsv'), sep='\t', index=False)
        self.test_df.to_csv(os.path.join(self.path, f'reshuffle_{self.seed}/test.tsv'), sep='\t', index=False)

    def _convert_to_internal_ids(self):
        self.user_mapping = pd.DataFrame(enumerate(self.train_df.user_id.unique()), columns=['remap_id', 'org_id'])
        self.item_mapping = pd.DataFrame(enumerate(self.train_df.asin.unique()), columns=['remap_id', 'org_id'])
        self.user_mapping = self.user_mapping.astype({'remap_id': int, 'org_id': str})
        self.item_mapping = self.item_mapping.astype({'remap_id': int, 'org_id': str})

        self.train_df.user_id = self.train_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.test_df.user_id = self.test_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.train_df.asin = self.train_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))
        self.test_df.asin = self.test_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))

    def _build_dicts(self):
        ''' build dicts for fast lookup '''
        self.n_users = self.train_df.user_id.nunique()
        self.n_items = self.train_df.asin.nunique()
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.bucket_len = self.n_train // self.n_users  # number of samples per user
        self.iterable_len = self.bucket_len * self.n_users  # length of torch Dataset we convert into
        self.all_items = range(self.n_items)  # this needs to be a python list for correct set conversion

        self.cached_samplings = defaultdict(list)
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].aggregate(list)
        self.positive_lists = [{
            'list': self.train_user_dict[u],  # list for faster random.choice
            'set': set(self.train_user_dict[u]),  # set for faster "x in y" check
            'tensor': torch.tensor(self.train_user_dict[u]).to(self.device),  # tensor for faster set difference
        } for u in range(self.n_users)]

        # split test into batches once at init instead of at every predict
        test_user_agg = self.test_df.groupby('user_id')['asin'].aggregate(list)

        # list of lists with test samples (per user), used for evaluation
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
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo().astype(np.float64)
        self.norm_matrix = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)

    def _adjacency_matrix(self):
        ''' create bipartite graph with initial vectors '''
        graph = dgl.heterograph({
            ('item', 'bought_by', 'user'): (self.train_df['asin'].values, self.train_df['user_id'].values),
            ('user', 'bought', 'item'): (self.train_df['user_id'].values, self.train_df['asin'].values),
        })
        user_ids = torch.tensor(list(range(self.n_users)), dtype=torch.long)
        item_ids = torch.tensor(range(self.n_items), dtype=torch.long)
        graph.ndata['id'] = {'user': user_ids, 'item': item_ids}
        return graph.adj_external(etype='bought', scipy_fmt='coo', ctx=self.device)

    def _convert_sp_mat_to_sp_tensor(self, coo):
        ''' convert sparse matrix into torch sparse tensor '''
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _print_info(self):
        self.logger.info(f"n_train:    {self.n_train:-7}")
        self.logger.info(f"n_test:     {self.test_df.shape[0]:-7}")
        self.logger.info(f"n_users:    {self.n_users:-7}")
        self.logger.info(f"n_items:    {self.n_items:-7}")

    ''' this is done for compatibility w torch Dataset class '''

    def __len__(self):
        return self.iterable_len

    def __getitem__(self, idx: int):
        '''
        to iterate over entire train_df per epoch, total length of the dataset should be n_train, not n_users
        hence each user has a continuous 'bucket' of self.n_train // self.n_users items,
        incoming idx is the id of the element. to find the id of the bucket (i.e. user), divide by its length
        '''
        idx //= self.bucket_len

        '''
        precaching pos and neg samples for users to save time during iteration
        sample exactly as many examples at the beginning of each epoch as we'll need per item
        '''
        if not self.cached_samplings[idx]:
            positives = random.choices(self.positive_lists[idx]['list'], k=self.bucket_len)
            neg_samples = set()
            while len(neg_samples) < self.bucket_len * self.neg_samples:
                neg_sample = random.choice(self.all_items)
                if neg_sample not in self.positive_lists[idx]['set']:
                    neg_samples.add(neg_sample)
            negatives = np.array(list(neg_samples)).reshape(-1, self.bucket_len)
            self.cached_samplings[idx] = deque(torch.tensor(list(zip(repeat(idx), positives, *negatives))))
        return self.cached_samplings[idx].pop()
