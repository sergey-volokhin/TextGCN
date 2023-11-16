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

    def __init__(self, args) -> None:
        self._copy_args(args)
        self._load_files(args.reshuffle, args.seed)
        self._convert_to_internal_ids()
        self._print_info()
        self._build_dicts()
        self._precalculate_normalization()

        assert all(
            self.n_items > i for i in args.k
        ), f'all values of k must be smaller than number of items ({self.n_items}), got k={args.k}'

    def _copy_args(self, args) -> None:
        self.path = args.data
        self.slurm = args.slurm
        self.logger = args.logger
        self.device = args.device
        self.batch_size = args.batch_size
        self.neg_samples = args.neg_samples

    def _load_files(self, reshuffle: bool, seed: int = 42) -> None:  # todo simplify loading
        self.logger.info('loading data')

        reshuffle_folder = os.path.join(self.path, f'reshuffle_{seed}')
        if reshuffle and os.path.exists(os.path.join(reshuffle_folder, 'train.tsv')):
            self.train_df = pd.read_table(os.path.join(reshuffle_folder, 'train.tsv'), dtype=str)
            self.test_df = pd.read_table(os.path.join(reshuffle_folder, 'test.tsv'), dtype=str)
        else:
            self.train_df = pd.read_table(self.path + 'train.tsv', dtype=str)
            self.test_df = pd.read_table(self.path + 'test.tsv', dtype=str)
            if reshuffle:
                self._reshuffle_train_test(seed=seed)

        ''' remove items from test that don't appear in train '''
        if set(self.test_df['asin'].unique()) - set(self.train_df['asin'].unique()):
            self.test_df = self.test_df[self.test_df['asin'].isin(self.train_df['asin'].unique())]
        ''' remove users from train that don't appear in test '''
        if set(self.test_df['user_id'].unique()) - set(self.train_df['user_id'].unique()):
            self.train_df = self.train_df[self.train_df['user_id'].isin(self.test_df['user_id'].unique())]

    def _reshuffle_train_test(self, train_size: float = 0.8, seed: int = 42) -> None:
        os.makedirs(os.path.join(self.path, f'reshuffle_{seed}'), exist_ok=True)

        train, test = [], []
        for user, group in tqdm(pd.concat([self.train_df, self.test_df]).groupby('user_id'),
                                desc='reshuffling',
                                dynamic_ncols=True,
                                leave=False,
                                disable=self.slurm):
            assert len(group) > 3, f'too few reviews for user {user}: {len(group)}'
            group = group.sample(frac=1, random_state=seed)
            group_train_size = round(group.shape[0] * train_size)
            train.append(group.iloc[:group_train_size])
            test.append(group.iloc[group_train_size:])

        self.train_df = pd.concat(train)
        self.test_df = pd.concat(test)

        ''' remove items from test that don't appear in train '''
        self.test_df = self.test_df[self.test_df['asin'].isin(self.train_df['asin'].unique())]
        ''' remove users from train that don't appear in test '''
        self.train_df = self.train_df[self.train_df['user_id'].isin(self.test_df['user_id'].unique())]

        self.train_df.to_csv(self.path + f'reshuffle_{seed}/train.tsv', sep='\t', index=False)
        self.test_df.to_csv(self.path + f'reshuffle_{seed}/test.tsv', sep='\t', index=False)

    def _convert_to_internal_ids(self) -> None:
        # remove users that do not appear in test_df
        user_not_in_train = set(self.train_df.user_id.unique()) - set(self.test_df.user_id.unique())
        self.train_df = self.train_df[~self.train_df.user_id.isin(user_not_in_train)]

        self.user_mapping = pd.DataFrame(enumerate(self.train_df.user_id.unique()), columns=['remap_id', 'org_id'])
        self.item_mapping = pd.DataFrame(enumerate(self.train_df.asin.unique()), columns=['remap_id', 'org_id'])

        self.user_mapping = self.user_mapping.astype({'remap_id': int, 'org_id': str})
        self.item_mapping = self.item_mapping.astype({'remap_id': int, 'org_id': str})

        self.train_df.user_id = self.train_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.test_df.user_id = self.test_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.train_df.asin = self.train_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))
        self.test_df.asin = self.test_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))

    def _build_dicts(self) -> None:
        self.cached_samplings = defaultdict(list)
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].aggregate(list)
        self.positive_lists = [
            {
                'list': self.train_user_dict[u],  # list for faster random.choice
                'set': set(self.train_user_dict[u]),  # set for faster "x in y" check
                'tensor': torch.tensor(self.train_user_dict[u]).to(self.device),  # tensor for faster set difference
            }
            for u in range(self.n_users)
        ]

        # split test into batches once at init instead of at every predict
        test_user_agg = self.test_df.groupby('user_id')['asin'].aggregate(list)
        users = test_user_agg.index.to_list()
        self.test_batches = [users[j:j + self.batch_size] for j in range(0, len(users), self.batch_size)]

        # list of lists with test samples (per user)
        self.true_test_lil = test_user_agg.values.tolist()

    def _precalculate_normalization(self) -> None:
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

    def _adjacency_matrix(self) -> dgl.DGLHeteroGraph:
        ''' create bipartite graph with initial vectors '''
        graph = dgl.heterograph(
            {
                ('item', 'bought_by', 'user'): (self.train_df['asin'].values, self.train_df['user_id'].values),
                ('user', 'bought', 'item'): (self.train_df['user_id'].values, self.train_df['asin'].values),
            }
        )
        user_ids = torch.tensor(list(range(self.n_users)), dtype=torch.long)
        item_ids = torch.tensor(range(self.n_items), dtype=torch.long)
        graph.ndata['id'] = {'user': user_ids, 'item': item_ids}
        return graph.adj_external(etype='bought', scipy_fmt='coo', ctx=self.device)

    def _convert_sp_mat_to_sp_tensor(self, coo) -> torch.Tensor:
        ''' convert sparse matrix into torch sparse tensor '''
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _print_info(self) -> None:
        self.n_users = self.train_df.user_id.nunique()
        self.n_items = self.train_df.asin.nunique()
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.bucket_len = self.n_train // self.n_users  # number of samples per user
        self.iterable_len = self.bucket_len * self.n_users  # length of torch Dataset we convert into
        self.all_items = range(self.n_items)  # this needs to be a python list for correct set conversion

        self.logger.info(f"n_train:    {self.n_train:-7}")
        self.logger.info(f"n_test:     {self.test_df.shape[0]:-7}")
        self.logger.info(f"n_users:    {self.n_users:-7}")
        self.logger.info(f"n_items:    {self.n_items:-7}")

    ''' this is done for compatibility w torch Dataset class '''

    def __len__(self) -> int:
        return self.iterable_len

    def __getitem__(self, idx: int):
        '''
        each user has a continuous 'bucket' of self.n_train // self.n_users items,
        incoming idx is the id of the element. to find the id of the bucket, divide by its length
        '''
        idx //= self.bucket_len

        '''
        precaching pos and neg samples for users to save time during iteration
        sample exactly as many examples at the beginning of each epoch as we'll need per item
        '''
        if not self.cached_samplings[idx]:
            positives = random.choices(self.positive_lists[idx]['list'], k=self.bucket_len)
            while True:
                negatives = random.choices(self.all_items, k=self.bucket_len * self.neg_samples)  # !VERY inefficient
                if len(set(negatives).intersection(self.positive_lists[idx]['set'])) == 0:
                    break
            negatives = [negatives[i:i + self.bucket_len] for i in range(0, len(negatives), self.bucket_len)]
            self.cached_samplings[idx] = deque(torch.tensor(list(zip(repeat(idx), positives, *negatives))))

        return self.cached_samplings[idx].pop()
