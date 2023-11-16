import torch
from torch import nn

from .base_model import BaseModel
from .kg_models import DatasetKG
from .reviews_models import DatasetReviews


class LTRDataset(DatasetKG, DatasetReviews):
    ''' combines KG and Reviews datasets '''

    def __init__(self, args):
        super().__init__(args)
        self._get_users_as_avg_reviews()
        self._get_users_as_avg_desc()

    def _get_users_as_avg_reviews(self):
        ''' use average of reviews to represent users '''
        user_text_embs = {}
        for user, group in self.top_med_reviews.groupby('user_id')['vector']:
            user_text_embs[user] = torch.tensor(group.values.tolist()).mean(axis=0)
        self.users_as_avg_reviews = torch.stack(self.user_mapping['remap_id'].map(
            user_text_embs).values.tolist()).to(self.device)

    def _get_users_as_avg_desc(self):
        ''' use mean of descriptions to represent users '''
        user_text_embs = {}
        for user, group in self.top_med_reviews.groupby('user_id')['asin']:
            user_text_embs[user] = self.items_as_desc[group.values].mean(axis=0).cpu()
        self.users_as_avg_desc = torch.stack(self.user_mapping['remap_id'].map(
            user_text_embs).values.tolist()).to(self.device)


class LTRBase(BaseModel):
    '''
    base class for Learning-to-Rank models
    uses pre-trained LightGCN vectors and trains a layer on top
    '''

    def _copy_args(self, args):
        super()._copy_args(args)
        self.load_base = args.load_base
        self.unfreeze = args.unfreeze

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.users_as_avg_desc = dataset.users_as_avg_desc
        self.items_as_desc = dataset.items_as_desc
        self.all_items = dataset.all_items

    def _init_embeddings(self, emb_size):
        super()._init_embeddings(emb_size)
        if not self.unfreeze:
            self.embedding_user.requires_grad_(False)
            self.embedding_item.requires_grad_(False)

    def _add_vars(self, args):
        super()._add_vars(args)

        ''' load the base model before overwriting the scoring functions '''
        if self.load_base:
            self.load_model(self.load_base)

        ''' features we are going to use'''
        self.feature_names = [
            'lightgcn score',
            'reviews',
            'desc',
            'reviews-description',
            'description-reviews',
        ]
        ''' build the trainable layer on top '''
        self._setup_layers(args)

    def _setup_layers(self, *args):
        ''' build the top predictive layer that takes features from LightGCN '''
        raise NotImplementedError

    def score_batchwise_ltr(self, *args):
        ''' analogue of score_batchwise in BaseModel that uses textual data '''
        raise NotImplementedError

    def score_pairwise_ltr(self, *args):
        ''' analogue of score_pairwise in BaseModel that uses textual data '''
        raise NotImplementedError

    ''' utilities: representation functions '''

    def get_user_reviews_mean(self, u):
        ''' represent users as mean of their reviews '''
        return self.users_as_avg_reviews[u]

    def get_user_desc(self, u):
        ''' represent users as mean of descriptions of items they reviewed '''
        return self.users_as_avg_desc[u]

    def get_item_reviews_mean(self, i, *args):
        ''' represent items as mean of their reviews '''
        return self.items_as_avg_reviews[i]

    def get_item_desc(self, i, *args):
        ''' represent items as their description '''
        return self.items_as_desc[i]

    def get_item_reviews_user(self, i, u):
        ''' represent items as the review of corresponding user '''
        df = self.reviews_df.loc[torch.stack([i, u], axis=1).tolist()]
        return torch.tensor(df.values.tolist()).to(self.device)

    def get_item_vectors(self, items_emb, items):
        ''' get vectors used to calculate textual representations dense features for items '''
        vecs = {'emb': items_emb}
        vecs['desc'] = self.get_item_desc(items)
        vecs['reviews'] = self.get_item_reviews_mean(items)
        return vecs

    def get_user_vectors(self, users_emb, users):
        ''' get vectors used to calculate textual representations dense features for users '''
        vecs = {'emb': users_emb}
        vecs['desc'] = self.get_user_desc(users)
        vecs['reviews'] = self.get_user_reviews_mean(users)
        return vecs

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


class LTRLinear(LTRBase):
    ''' trains a dense layer on top of LightGCN '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        ''' overwrite the scoring methods '''
        self.evaluate = self.evaluate_ltr
        self.score_pairwise = self.score_pairwise_ltr
        self.score_batchwise = self.score_batchwise_ltr

    def _setup_layers(self, args):
        '''
        dense layers that combine all the scores from different node representations
        layer_sizes: represents the size and number of hidden layers (default: no hidden layers)
        '''
        layer_sizes = [len(self.feature_names)] + args.ltr_layers + [1]
        layers = []
        for i, j in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(i, j))
        self.layers = nn.Sequential(*layers).to(self.device)

    def evaluate_ltr(self):
        ''' print weights (i.e. feature importances) if the model consists of single layer '''
        if len(self.layers) == 1:
            self.logger.info('Feature weights from the top layer:')
            for f, w in zip(self.feature_names, self.layers[0].weight.tolist()[0]):
                self.logger.info(f'{f:<20} {w:.4}')
        return super().evaluate()

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        return self.layers(self.get_features_batchwise(u_vecs, i_vecs)).squeeze()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, items)
        return self.layers(self.get_features_pairwise(u_vecs, i_vecs))


class LTRLinearWPop(LTRLinear):
    ''' extends LTRLinear by adding popularity features '''

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.popularity_users = dataset.popularity_users
        self.popularity_items = dataset.popularity_items

    def _setup_layers(self, args):
        self.feature_names += ['user popularity', 'item popularity']
        super()._setup_layers(args)

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        features = self.get_features_batchwise(u_vecs, i_vecs)
        pop_u = self.popularity_users[users].unsqueeze(-1).expand(len(users), self.n_items, 1)
        pop_i = self.popularity_items.expand(len(users), self.n_items, 1)
        return self.layers(torch.cat([features, pop_u, pop_i], axis=-1)).squeeze()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, items)
        features = self.get_features_pairwise(u_vecs, i_vecs)
        return self.layers(torch.cat([
            features,
            self.popularity_users[users],
            self.popularity_items[items],
        ], axis=-1))
