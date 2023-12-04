from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

from .BaseModel import BaseModel
from .kg_models import DatasetKG
from .LightGCN import LightGCN
from .reviews_models import DatasetReviews


class LTRDataset(DatasetKG, DatasetReviews):
    ''' combines KG and Reviews datasets '''


class LTRBase(BaseModel, ABC):
    '''
    base class for Learning-to-Rank models
    uses pre-trained LightGCN vectors and trains something on top
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
        self.LightGCN = LightGCN(params, dataset)
        if params.load_base:
            self.logger.info(f'Loading base LightGCN model from {params.load_base}.')
            if not params.freeze:
                self.logger.warn('Base model not frozen for LTR model, this will degrade performance')
            self.LightGCN.load(params.load_base)
            base_results = self.LightGCN.evaluate()
            self.print_metrics(base_results)
        else:
            self.logger.warn('Not using a pretrained base model leads to poor performance.')

    def _add_vars(self, *args, **kwargs):
        super()._add_vars(*args, **kwargs)

        self.activation = F.selu  # F.softmax

        ''' features we are going to use'''
        self.feature_names = [
            'lightgcn score',
            'reviews',
            'desc',
            'reviews-description',
            'description-reviews',
        ]

    def get_loss(self, data):
        users, pos, *negs = data.to(self.device).t()
        bpr_loss = self.bpr_loss(users, pos, negs)
        reg_loss = self.LightGCN.reg_loss(users, pos, negs)
        self._loss_values['bpr'] += bpr_loss
        self._loss_values['reg'] += reg_loss
        return bpr_loss + reg_loss

    def bpr_loss(self, users, pos, negs):
        ''' Bayesian Personalized Ranking pairwise loss '''
        users_emb, items_emb = self.representation
        users_emb = users_emb[users]
        pos_scores = self.score_pairwise(users_emb, items_emb[pos], users, pos)
        loss = 0
        for neg in negs:  # todo: vectorize
            neg_scores = self.score_pairwise(users_emb, items_emb[neg], users, neg)
            loss += torch.mean(self.activation(neg_scores - pos_scores))
        loss /= len(negs)
        return loss

    @property
    def representation(self):
        return self.LightGCN.representation

    @abstractmethod
    def _setup_layers(self, *args, **kwargs):
        ''' build the top predictive layer that takes features from LightGCN '''

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


class LTRLinear(LTRBase, ABC):
    ''' trains a dense layer on top of LightGCN '''

    def _setup_layers(self, params):
        '''
        dense layers that combine all the scores from different node representations
        layer_sizes: represents the size and number of hidden layers (default: no hidden layers)
        '''
        layer_sizes = [len(self.feature_names)] + params.ltr_layers + [1]
        layers = [nn.Linear(i, j) for i, j in zip(layer_sizes, layer_sizes[1:])]
        self.layers = nn.Sequential(*layers).to(self.device)

    def evaluate(self, *args, **kwargs):
        ''' print weights (i.e. feature importances) if the model consists of single layer '''
        if len(self.layers) == 1:
            self.logger.info('Feature weights from the top layer:')
            for f, w in zip(self.feature_names, self.layers[0].weight.tolist()[0]):
                self.logger.info(f'{f:<20} {w:.4}')
        return super().evaluate(*args, **kwargs)

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


class LTRLinearWPop(LTRLinear):
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
