from abc import ABC, abstractmethod

import torch
from torch import nn

from .BaseModel import BaseModel
from .DatasetKG import DatasetKG
from .DatasetReviews import DatasetReviews
from .DatasetRatings import DatasetRatings


class LTRDatasetRank(DatasetKG, DatasetReviews):
    ''' combines KG and Reviews datasets '''


class LTRDatasetScore(DatasetKG, DatasetReviews, DatasetRatings):
    ''' combines KG, Reviews and Ratings datasets '''


class LTRBaseModel(BaseModel, ABC):
    '''
    Objective-agnostic model that joins GCN model
    with textual features (reviews and descriptions)
    trains something on top
    '''

    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self._setup_layers(params)
        self._load_base(params, dataset)

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.users_as_avg_desc = dataset.users_as_avg_desc
        self.items_as_desc = dataset.items_as_desc
        self.all_items = dataset.all_items

    def _load_base(self, params, dataset):
        ''' load the base model '''
        self.foundation = self.foundation_class(params, dataset)
        if params.load_base:
            self.logger.info(f'Loading base LightGCN model from {params.load_base}.')
            if not params.freeze:
                self.logger.warn('Base model not frozen for LTR model, this will degrade performance')
            self.foundation.load(params.load_base)
            results = self.foundation.evaluate()
            self.metrics_log.update(results)
            self.logger.info('Base model metrics:')
            self.metrics_log.log()
        else:
            self.logger.warn('Not using a pretrained base model leads to poor performance.')

    def _add_vars(self, *args, **kwargs):
        super()._add_vars(*args, **kwargs)

        ''' features we are going to use'''
        self.feature_names = [
            'base_model_score',
            'reviews',
            'desc',
            'reviews-description',
            'description-reviews',
        ]

    def _setup_layers(self, params):
        '''
        dense layers that combine all the scores from different node representations
        layer_sizes: represents the size and number of hidden layers (default: no hidden layers)
        '''
        layer_sizes = [len(self.feature_names)] + params.ltr_layers + [1]
        layers = [nn.Linear(i, j) for i, j in zip(layer_sizes, layer_sizes[1:])]
        self.layers = nn.Sequential(*layers).to(self.device)

    def reg_loss(self, *args, **kwargs):
        return self.foundation.reg_loss(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.foundation.forward(*args, **kwargs)

    def score_batchwise(self, users_emb, items_emb, users):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        features = self.get_features_batchwise(u_vecs, i_vecs)
        return self.layers(features).squeeze()

    def score_pairwise(self, users_emb, items_emb, users, items):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, items)
        features = self.get_features_pairwise(u_vecs, i_vecs)
        return self.layers(features)

    def evaluate(self, *args, **kwargs):
        ''' print weights (i.e. feature importances) if the model consists of single layer '''
        if len(self.layers) == 1:
            self.logger.info('Feature weights from the top layer:')
            for f, w in zip(self.feature_names, self.layers[0].weight.tolist()[0]):
                self.logger.info(f'{f:<20} {w:.4}')
        return super().evaluate(*args, **kwargs)

    ''' representation functions '''

    def get_user_reviews_mean(self, u):
        ''' represent users as mean of their reviews '''
        return self.users_as_avg_reviews[u]

    def get_user_desc(self, u):
        ''' represent users as mean of descriptions of items they reviewed '''
        return self.users_as_avg_desc[u]

    def get_item_reviews_mean(self, i):
        ''' represent items as mean of their reviews '''
        return self.items_as_avg_reviews[i]

    def get_item_desc(self, i):
        ''' represent items as their description '''
        return self.items_as_desc[i]

    def get_item_reviews_user(self, i, u):
        ''' represent items as the review of corresponding user '''
        df = self.reviews_vectors.loc[torch.stack([i, u], axis=1).tolist()]
        return torch.tensor(df.values.tolist()).to(self.device)

    def get_item_vectors(self, items_emb, items):
        ''' get vectors used to calculate textual representations dense features for items '''
        return {
            'emb': items_emb,
            'desc': self.get_item_desc(items),
            'reviews': self.get_item_reviews_mean(items),
        }

    def get_user_vectors(self, users_emb, users):
        ''' get vectors used to calculate textual representations dense features for users '''
        return {
            'emb': users_emb,
            'desc': self.get_user_desc(users),
            'reviews': self.get_user_reviews_mean(users),
        }

    # todo get_scores_batchwise and get_scores_pairwise return scores that differ by 1e-5. why?
    def get_features_batchwise(self, u_vecs, i_vecs):
        '''
        batchwise (all-to-all) calculation of features for top layer:
        vectors['users_emb'].shape = (batch_size, emb_size)
        vectors['items_emb'].shape = (n_items, emb_size)
        '''
        return torch.cat(
            [
                (u_vecs['emb'] @ i_vecs['emb'].T).unsqueeze(-1),
                (u_vecs['reviews'] @ i_vecs['reviews'].T).unsqueeze(-1),
                (u_vecs['desc'] @ i_vecs['desc'].T).unsqueeze(-1),
                (u_vecs['reviews'] @ i_vecs['desc'].T).unsqueeze(-1),
                (u_vecs['desc'] @ i_vecs['reviews'].T).unsqueeze(-1),
            ],
            axis=-1,
        )

    def get_features_pairwise(self, u_vecs, i_vecs):
        '''
        pairwise (1-to-1) calculation of features for top layer:
        vectors['users_emb'].shape == vectors['items_emb'].shape
        '''

        def sum_mul(x, y):
            return (x * y).sum(dim=1).unsqueeze(1)

        return torch.cat(
            [
                sum_mul(u_vecs['emb'], i_vecs['emb']),
                sum_mul(u_vecs['reviews'], i_vecs['reviews']),
                sum_mul(u_vecs['desc'], i_vecs['desc']),
                sum_mul(u_vecs['reviews'], i_vecs['desc']),
                sum_mul(u_vecs['desc'], i_vecs['reviews']),
            ],
            axis=1,
        )


class LTRBaseWPop(LTRBaseModel):
    ''' extends LTRLinear by adding popularity features '''

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.popularity_users = dataset.popularity_users
        self.popularity_items = dataset.popularity_items

    def _add_vars(self, *args, **kwargs):
        super()._add_vars(*args, **kwargs)
        self.feature_names += ['user popularity', 'item popularity']

    def score_batchwise(self, users_emb, items_emb, users):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        features = self.get_features_batchwise(u_vecs, i_vecs)
        pop_u = self.popularity_users[users].unsqueeze(-1).expand(len(users), self.n_items, 1)
        pop_i = self.popularity_items.expand(len(users), self.n_items, 1)
        return self.layers(torch.cat([features, pop_u, pop_i], axis=-1)).squeeze()

    def score_pairwise(self, users_emb, items_emb, users, items):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, items)
        features = self.get_features_pairwise(u_vecs, i_vecs)
        return self.layers(torch.cat([
            features,
            self.popularity_users[users],
            self.popularity_items[items],
        ], axis=-1))
