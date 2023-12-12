import os
import random

import pandas as pd
import torch

from .BaseDataset import BaseDataset


class DatasetRatings(BaseDataset):

    def __init__(self, config):
        super().__init__(config)
        self._load_ratings()
        self._normalize_ratings()

    def _load_ratings(self):
        ''' add ratings to train, val and test '''
        if 'rating' in self.train_df.columns:
            self.logger.info('Ratings already in train_df')
            self.train_df.rating = self.train_df.rating.astype(float).astype(int)
            self.test_df.rating = self.test_df.rating.astype(float).astype(int)
            if hasattr(self, '_actual_test_df'):
                self._actual_test_df.rating = self._actual_test_df.rating.astype(float).astype(int)
            return

        self.logger.info('Loading ratings')
        ratings = pd.read_table(
            os.path.join(self.path, 'reviews_text.tsv'),
            usecols=['asin', 'user_id', 'rating'],
            dtype=str,
        )
        overall = ratings.set_index(['asin', 'user_id'])['rating'].astype(float).astype(int)

        train_indexed = self.train_df.set_index(['asin', 'user_id'])
        train_indexed['rating'] = overall
        self.train_df = train_indexed.reset_index()

        test_indexed = self.test_df.set_index(['asin', 'user_id'])
        test_indexed['rating'] = overall
        self.test_df = test_indexed.reset_index()

        if hasattr(self, '_actual_test_df'):
            actual_test_indexed = self._actual_test_df.set_index(['asin', 'user_id'])
            actual_test_indexed['rating'] = overall
            self._actual_test_df = actual_test_indexed.reset_index()

    def _normalize_ratings(self):
        ''' normalize ratings user-wise using sklearn scaler '''

        self.logger.info('Normalizing ratings')

        groupped = self.train_df.groupby('user_id')
        user_means = groupped['rating'].transform('mean')
        user_stds = groupped['rating'].transform('std').fillna(0)
        self.train_df['normal_rating'] = ((self.train_df['rating'] - user_means) / user_stds).fillna(0)

        groupped = self.train_df.groupby('user_id')
        scalers = groupped.agg({'rating': ['mean', 'std']}).fillna(0)
        scalers.columns = scalers.columns.droplevel()
        self.scalers = scalers.to_dict(orient='index')

        self.train_df_user_groupped = {  # for faster __getitem__
            user_id: data[['user_id', 'asin', 'normal_rating']].values
            for user_id, data in groupped
        }

    def __getitem__(self, idx: int):
        '''
        sample a single random scored item
        returns a torch with shape (3,): [idx, item_id, score]
        '''
        idx //= self.bucket_len
        user_data = self.train_df_user_groupped[idx]
        return torch.from_numpy(user_data[random.randint(0, len(user_data) - 1)]).to(self.device)
