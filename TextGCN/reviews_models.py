import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .dataset import BaseDataset
from .text_base_model import TextBaseModel
from .utils import embed_text
import time


class DatasetReviews(BaseDataset):

    def __init__(self, params):
        super().__init__(params)
        self._load_reviews()
        self._calc_review_embs(params.emb_batch_size, params.bert_model)
        self._get_items_as_avg_reviews()
        self._calc_popularity()

    def _load_reviews(self):
        self.reviews = pd.read_table(os.path.join(self.path, 'reviews_text.tsv'), dtype=str)
        if 'time' not in self.reviews.columns:
            self.reviews['time'] = 0
        self.reviews = self.reviews[['asin', 'user_id', 'review', 'time']].sort_values(['asin', 'user_id'])
        self.reviews.user_id = self.reviews.user_id.map(dict(self.user_mapping[['org_id', 'remap_id']].values))
        self.reviews.asin = self.reviews.asin.map(dict(self.item_mapping[['org_id', 'remap_id']].values))
        self.reviews = self.reviews.dropna()
        self.reviews[['asin', 'user_id']] = self.reviews[['asin', 'user_id']].astype(int)

    def _calc_review_embs(
        self,
        emb_batch_size: int,
        bert_model: str,
    ):
        ''' load/calc embeddings of the reviews and setup the dicts '''
        emb_file = os.path.join(
            self.path,
            'embeddings',
            f'item_full_reviews_loss_repr_{bert_model.split("/")[-1]}.torch',
        )
        self.reviews['vector'] = (
            embed_text(
                self.reviews['review'],
                emb_file,
                bert_model,
                emb_batch_size,
                self.device,
            )
            .cpu()
            .numpy()
            .tolist()
        )

        ''' dropping testset reviews '''
        # doing it here, not at loading, to not recalculate textual embs if resplitting train-test
        reviews_indexed = self.reviews.set_index(['asin', 'user_id'])
        test_indexed = self.test_df.set_index(['asin', 'user_id'])
        self.reviews = self.reviews[~reviews_indexed.index.isin(test_indexed.index)]
        self.reviews_vectors = self.reviews.set_index(['asin', 'user_id'])['vector']

    def _get_items_as_avg_reviews(self):
        ''' use average of reviews to represent items '''

        # number of reviews to use for representing items and users
        num_reviews = int(
            np.median(
                pd.concat([self.reviews.groupby('asin')['user_id'].size(),
                           self.reviews.groupby('user_id')['asin'].size()])
            )
        )

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

        item_text_embs = {}
        for item, group in self.top_med_reviews.groupby('asin')['vector']:
            item_text_embs[item] = torch.tensor(group.values.tolist()).mean(axis=0)
        self.items_as_avg_reviews = self.item_mapping['remap_id'].map(item_text_embs).values.tolist()
        self.items_as_avg_reviews = torch.stack(self.items_as_avg_reviews).to(self.device)

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


class TextModelReviews(TextBaseModel):

    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        ''' how do we textually represent items in sampled triplets '''
        if params.pos == 'avg' or params.model == 'reviews':
            self.get_pos_items_reprs = self.get_item_reviews_mean
        elif params.pos == 'user':
            self.get_pos_items_reprs = self.get_item_reviews_user

        if params.neg == 'avg' or params.model == 'reviews':
            self.get_neg_items_reprs = self.get_item_reviews_mean

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.reviews_vectors = dataset.reviews_vectors
        self.items_as_avg_reviews = dataset.items_as_avg_reviews

    def get_item_reviews_mean(self, items):
        ''' represent items with mean of their reviews '''
        return self.items_as_avg_reviews[items]

    def get_item_reviews_user(self, items, users):
        ''' represent items with the review of corresponding user '''
        df = self.reviews_vectors.loc[torch.stack([items, users], axis=1).tolist()]
        return torch.from_numpy(df.values).to(self.device)
