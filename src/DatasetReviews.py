import os

import numpy as np
import pandas as pd
import torch

from .BaseDataset import BaseDataset
from .utils import embed_text


class DatasetReviews(BaseDataset):
    '''
    Dataset with textual reviews
    calculates textual representations of items and users using reviews
    also calculates popularity of items and users
    '''

    def _load_reviews(self, config):
        self._load_review_file()
        self._cut_reviews_to_median()
        self._calc_review_embs(model_name=config.encoder, emb_batch_size=config.emb_batch_size)
        self._get_items_as_avg_reviews()
        self._get_users_as_avg_reviews()
        # self._calc_popularity()

    def _load_review_file(self):
        self.reviews = pd.read_table(os.path.join(self.path, 'reviews_text.tsv'), dtype=str)
        if 'time' not in self.reviews.columns:
            self.reviews['time'] = 0
        self.reviews = self.reviews[['asin', 'user_id', 'review', 'time']].sort_values(['asin', 'user_id'])
        self.reviews.user_id = self.reviews.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.reviews.asin = self.reviews.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))
        self.reviews = self.reviews.dropna()
        self.reviews[['asin', 'user_id']] = self.reviews[['asin', 'user_id']].astype(int)

        ''' dropping testset reviews '''
        reviews_indexed = self.reviews.set_index(['asin', 'user_id'])
        test_indexed = self.test_df.set_index(['asin', 'user_id'])
        reviews_indexed.drop(reviews_indexed.index.intersection(test_indexed.index), inplace=True)
        if hasattr(self, 'val_df'):
            val_indexed = self.val_df.set_index(['asin', 'user_id'])
            reviews_indexed.drop(reviews_indexed.index.intersection(val_indexed.index), inplace=True)
        self.reviews = reviews_indexed.reset_index()

    def _cut_reviews_to_median(self):
        # number of reviews to use for representing items and users
        num_reviews = int(np.median(
            pd.concat([self.reviews.groupby('asin')['user_id'].size(),
                       self.reviews.groupby('user_id')['asin'].size()])
        ))
        self.logger.info(f'using {num_reviews} reviews for each item and user representation')

        # use only most recent reviews for representation
        top_reviews_by_user = (
            self.reviews.sort_values(by=['user_id', 'time'], ascending=[True, False])
            .groupby('user_id')
            .head(num_reviews)
        )
        top_reviews_by_item = (
            self.reviews.sort_values(by=['asin', 'time'], ascending=[True, False])
            .groupby('asin')
            .head(num_reviews)
        )

        # saving top_med_reviews to model so we could extend LTR
        self.top_med_reviews = (
            pd.concat([top_reviews_by_user, top_reviews_by_item])
            .drop_duplicates(subset=['asin', 'user_id'])
            .sort_values(['asin', 'user_id'])
            .reset_index(drop=True)
        )

    def _calc_review_embs(self, model_name: str, emb_batch_size: int = 64):
        self.logger.debug('getting review embeddings')
        ''' load/calc embeddings of the reviews and setup the dicts '''
        emb_file = os.path.join(
            self.path,
            'embeddings',
            f'item_full_reviews_loss_repr_{model_name.split("/")[-1]}.pkl',
        )
        self.top_med_reviews['vector'] = (
            embed_text(
                sentences=self.top_med_reviews['review'].tolist(),
                path=emb_file,
                model_name=model_name,
                batch_size=emb_batch_size,
                logger=self.logger,
                device=self.device,
            )
            .cpu()
            .numpy()
            .tolist()
        )

        self.text_emb_size = len(self.top_med_reviews['vector'].iloc[0])


    def _calc_popularity(self):
        ''' calculates normalized popularity of users and items, based on the number of reviews they have '''

        lengths = self.reviews.groupby('user_id')[['asin']].size().sort_values(ascending=False)
        self.popularity_users = (
            torch.tensor(lengths.reset_index()['user_id'].values / lengths.shape[0], dtype=torch.float)
            .to(self.device)
            .unsqueeze(1)
        )
        lengths = self.reviews.groupby('asin')[['user_id']].size().sort_values(ascending=False)
        self.popularity_items = (
            torch.tensor(lengths.reset_index()['asin'].values / lengths.shape[0], dtype=torch.float)
            .to(self.device)
            .unsqueeze(1)
        )

    def _get_items_as_avg_reviews(self):
        ''' use average of reviews to represent items '''
        item_text_embs = {}
        for item, group in self.top_med_reviews.groupby('asin')['vector']:
            item_text_embs[item] = torch.tensor(group.values.tolist()).mean(axis=0)
        mapped = self.item_mapping['remap_id'].map(item_text_embs).values.tolist()
        mapped = [torch.zeros(self.text_emb_size) if isinstance(x, float) else x for x in mapped]
        self.item_representations['reviews'] = torch.stack(mapped).to(self.device)

    def _get_users_as_avg_reviews(self):
        ''' use average of reviews to represent users '''
        user_text_embs = {}
        for user, group in self.top_med_reviews.groupby('user_id')['vector']:
            user_text_embs[user] = torch.tensor(group.values.tolist()).mean(axis=0)

        mapped = self.user_mapping['remap_id'].map(user_text_embs).values.tolist()
        mapped = [torch.zeros(self.text_emb_size) if isinstance(x, float) else x for x in mapped]
        self.user_representations['reviews'] = torch.stack(mapped).to(self.device)
