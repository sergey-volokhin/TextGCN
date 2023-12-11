import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

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

        def sklearn_scale(rating):
            scaler = StandardScaler()
            scores = scaler.fit_transform(rating.values)
            return scores, {'scale': scaler.scale_[0], 'mean': scaler.mean_[0]}

        tqdm.pandas(desc='normalizing ratings', dynamic_ncols=True, disable=self.slurm)
        normalized_scores, scalers = zip(*self.train_df.groupby('user_id')[['rating']].progress_apply(sklearn_scale))
        self.train_df['normal_rating'] = np.concatenate(normalized_scores)
        self.scalers = dict(zip(self.train_df['user_id'].unique(), scalers))

        self.train_df_user_groupped = {  # for faster __getitem__
            group: data[['user_id', 'asin', 'normal_rating']].values
            for group, data in self.train_df.groupby('user_id')
        }

    def __getitem__(self, idx: int):
        '''
        sample a single random scored item
        returns a torch with shape (3,): [idx, item_id, score]
        '''
        idx //= self.bucket_len
        user_data = self.train_df_user_groupped[idx]
        return torch.from_numpy(user_data[random.randint(0, len(user_data) - 1)]).to(self.device)
