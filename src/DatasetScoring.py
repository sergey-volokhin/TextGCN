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

    def _load_files(self, *args, **kwargs):
        self.objective = 'scoring'
        super()._load_files(*args, **kwargs)

    def _train_test_split(self, df, train_size=0.8):
        self.train_df, self.val_df, self.test_df = ttss(df, train_size=train_size, seed=self.seed)

    def _load_ratings(self):
        '''add ratings to train, val and test'''
        if 'rating' in self.train_df.columns:
            self.logger.debug('ratings already in train_df')
            self.train_df.rating = self.train_df.rating.astype(float).astype(int)
            self.test_df.rating = self.test_df.rating.astype(float).astype(int)
            self.val_df.rating = self.val_df.rating.astype(float).astype(int)
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
        test_indexed = self.test_df.set_index(['asin', 'user_id'])
        val_indexed = self.val_df.set_index(['asin', 'user_id'])

        train_indexed['rating'] = ratings
        test_indexed['rating'] = ratings
        val_indexed['rating'] = ratings

        self.train_df = train_indexed.reset_index()
        self.test_df = test_indexed.reset_index()
        self.val_df = val_indexed.reset_index()

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
