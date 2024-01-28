import random
from os.path import join

import pandas as pd
import torch

from .BaseDataset import BaseDataset
from .utils import train_test_split_stratified as ttss


class DatasetScoring(BaseDataset):
    '''dataset for scoring task, includes ratings and validation set'''

    def __init__(self, config):
        super().__init__(config)
        self._load_ratings()
        self._normalize_ratings()

    ''' overwrite functions from BaseDataset first '''

    def _safety_checks(self):
        super()._safety_checks()
        items_not_in_train = set(self.test_df['asin'].unique()) - set(self.train_df['asin'].unique())
        assert not items_not_in_train, f"test set contains items that are not in the train set: {items_not_in_train}"
        users_not_in_train = set(self.test_df['user_id'].unique()) - set(self.train_df['user_id'].unique())
        assert not users_not_in_train, f"test set contains users that are not in the train set: {users_not_in_train}"

    def _load_files(self, reshuffle: bool):
        super()._load_files(reshuffle)

        self.logger.debug('loading validation set')
        self._actual_test_df = self.test_df.copy()  # hack to use validation set
        self.test_df = (
            pd.read_table(join(self.path, 'valid.tsv'), dtype=str)
            .sort_values(by=['user_id', 'asin'])
            .reset_index(drop=True)
        )

    def _convert_to_internal_ids(self):
        super()._convert_to_internal_ids()
        self._actual_test_df.user_id = self._actual_test_df.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self._actual_test_df.asin = self._actual_test_df.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))

    def _build_dicts(self):
        super()._build_dicts()
        # self.n_test = self.test_df.shape[0]
        self.n_val = self.test_df.shape[0]
        self.logger.info(f"n_val:      {self.n_val:-7}")

    def _reshuffle_train_test(self):
        super()._reshuffle_train_test()
        self._actual_test_df = self._actual_test_df.sort_values(by=['user_id', 'asin']).reset_index(drop=True)
        self._actual_test_df = self._actual_test_df[self._actual_test_df['asin'].isin(self.train_df['asin'].unique())]
        self._actual_test_df.to_csv(join(self.path, f'reshuffle_{self.seed}/test.tsv'), sep='\t', index=False)

    def _train_test_split(self, df, train_size=0.8):
        self.train_df, self.test_df, self._actual_test_df = ttss(df, train_size=train_size, seed=self.seed)

    ''' add functions specific to this dataset '''

    def _load_ratings(self):
        '''add ratings to train, val and test'''
        if 'rating' in self.train_df.columns:
            self.logger.debug('ratings already in train_df')
            self.train_df.rating = self.train_df.rating.astype(float).astype(int)
            self.test_df.rating = self.test_df.rating.astype(float).astype(int)
            self._actual_test_df.rating = self._actual_test_df.rating.astype(float).astype(int)
            return

        self.logger.debug('loading ratings')
        ratings = (
            pd.read_table(
                join(self.path, 'reviews_text.tsv'),
                usecols=['asin', 'user_id', 'rating'],
                dtype=str,
            )
            .set_index(['asin', 'user_id'])['rating']
            .astype(float)
            .astype(int)
        )

        train_indexed = self.train_df.set_index(['asin', 'user_id'])
        val_indexed = self.test_df.set_index(['asin', 'user_id'])
        test_indexed = self._actual_test_df.set_index(['asin', 'user_id'])

        train_indexed['rating'] = ratings
        val_indexed['rating'] = ratings
        test_indexed['rating'] = ratings

        self.train_df = train_indexed.reset_index()
        self.test_df = val_indexed.reset_index()
        self._actual_test_df = test_indexed.reset_index()

    def _normalize_ratings(self):
        '''normalize ratings user-wise by mean and std'''

        self.logger.debug('normalizing ratings')

        groupped = self.train_df.groupby('user_id')
        user_means = groupped['rating'].transform('mean')
        user_stds = groupped['rating'].transform('std').fillna(0)
        self.train_df['normal_rating'] = ((self.train_df['rating'] - user_means) / user_stds).fillna(0)

        groupped = self.train_df.groupby('user_id')
        scalers = groupped.agg({'rating': ['mean', 'std']}).fillna(0)
        scalers.columns = scalers.columns.droplevel()
        self.scalers = scalers.to_dict(orient='index')

        self.train_df_user_groupped = {  # for faster __getitem__
            user_id: data.values for user_id, data in groupped[['user_id', 'asin', 'normal_rating']]
        }

    def __getitem__(self, idx: int):
        '''
        sample a single random scored item
        returns a torch with shape (3,): [idx, item_id, score]
        '''
        idx //= self.bucket_len
        user_data = self.train_df_user_groupped[idx]
        return torch.from_numpy(user_data[random.randint(0, len(user_data) - 1)]).to(self.device)
