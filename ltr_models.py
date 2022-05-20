import torch
from torch import nn
from torch.nn import functional as F

from base_model import BaseModel
from kg_models import DatasetKG
from reviews_models import DatasetReviews
from sklearn.ensemble import GradientBoostingRegressor as GBRT


class LTRDataset(DatasetKG, DatasetReviews):

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
        base class for LTR models, which uses pre-trained
        LightGCN vectors to train a layer on top
    '''

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.users_as_avg_desc = dataset.users_as_avg_desc
        self.items_as_desc = dataset.items_as_desc

    def score_batchwise_ltr(self):
        ''' how do we predict the score using text data? '''
        raise NotImplementedError

    ''' utilities: representation functions '''

    def get_user_reviews_mean(self, u):
        ''' represent users with mean of their reviews '''
        return self.users_as_avg_reviews[u]

    def get_user_desc(self, u):
        ''' represent users as mean of descriptions of items they reviewed '''
        return self.users_as_avg_desc[u]

    def get_item_reviews_mean(self, i, u=None):
        ''' represent items with mean of their reviews '''
        return self.items_as_avg_reviews[i]

    def get_item_desc(self, i, u=None):
        ''' represent items with their description '''
        return self.items_as_desc[i]

    def get_item_reviews_user(self, i, u=None):
        ''' represent items with the review of corresponding user '''
        df = self.reviews_df.loc[torch.stack([i, u], axis=1).tolist()]
        return torch.tensor(df.values.tolist()).to(self.device)


class LTRLinear(LTRBase):
    ''' trains a dense layer on top of GNN '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self._setup_layers()
        self.score_pairwise = self.score_pairwise_ltr
        self.score_batchwise = self.score_batchwise_ltr
        self.evaluate = self.evaluate_ltr

    def _add_vars(self, args):
        super()._add_vars(args)
        self.dense_features = ['gnn', 'reviews', 'desc', 'rev_desc', 'desc_rev', 'gnn_rev', 'gnn_desc']
        if args.freeze_embeddings:
            self.embedding_user.requires_grad_(False)
            self.embedding_item.requires_grad_(False)

    def _setup_layers(self):
        ''' dense layer that combines all the scores from different node representations '''
        self.linear = nn.Linear(len(self.dense_features), 1).to(self.device)

    def get_scores(self, users_emb, items_emb, users, items):
        users_reviews = self.get_user_reviews_mean(users)
        items_reviews = self.get_item_reviews_mean(items)
        users_desc = self.get_user_desc(users)
        items_desc = self.get_item_desc(items)

        user_gnn_rev = torch.cat([users_emb, users_reviews], axis=1)
        item_gnn_rev = torch.cat([items_emb, items_reviews], axis=1)

        user_gnn_desc = torch.cat([users_emb, users_desc], axis=1)
        item_gnn_desc = torch.cat([items_emb, items_desc], axis=1)

        return torch.cat([  # todo find a shorthand replacement? this is non-normalized cos sim
            torch.sum(torch.mul(users_emb, items_emb), dim=1).unsqueeze(1),          # gnn-gnn
            torch.sum(torch.mul(users_reviews, items_reviews), dim=1).unsqueeze(1),  # reviews - reviews
            torch.sum(torch.mul(users_desc, items_desc), dim=1).unsqueeze(1),        # desc - desc
            torch.sum(torch.mul(users_reviews, items_desc), dim=1).unsqueeze(1),     # reviews - desc
            torch.sum(torch.mul(users_desc, items_reviews), dim=1).unsqueeze(1),     # desc - reviews
            torch.sum(torch.mul(user_gnn_rev, item_gnn_rev), dim=1).unsqueeze(1),    # gnn||reviews - gnn||reviews
            torch.sum(torch.mul(user_gnn_desc, item_gnn_desc), dim=1).unsqueeze(1),  # gnn||desc - gnn||desc
        ], axis=1)

    def score_pairwise_ltr(self, users_emb, items_emb, users, items, pos_or_neg):
        return self.linear(self.get_scores(users_emb, items_emb, users, items))

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        rating = []
        for user, user_emb in zip(users, users_emb):
            user_emb = user_emb.expand((items_emb.shape[0], user_emb.shape[0]))
            user = torch.tensor(user).expand((items_emb.shape[0]))
            scores = self.get_scores(user_emb, items_emb, user, list(range(self.n_items)))
            rating.append(self.linear(scores).squeeze())
        return torch.stack(rating)

    def evaluate_ltr(self):
        res = super().evaluate()
        self.logger.info('Feature weights:')
        for num, feat in zip(self.linear.weight.round(decimals=4).tolist()[0], self.dense_features):
            self.logger.info(f'{feat:<8} {num}')
        return res


class LTRLinearWPop(LTRLinear):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.popularity_users = dataset.popularity_users
        self.popularity_items = dataset.popularity_items

    def _add_vars(self, args):
        super()._add_vars(args)
        self.dense_features += ['pop_u', 'pop_i']

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        rating = []
        for user, user_emb in zip(users, users_emb):
            user_emb = user_emb.expand((items_emb.shape[0], user_emb.shape[0]))
            user = torch.tensor(user).expand((items_emb.shape[0]))
            scores = self.get_scores(user_emb, items_emb, user, list(range(self.n_items)))
            pop_u = self.popularity_users[user].expand((items_emb.shape[0],))
            pop_i = self.popularity_items
            rating.append(self.linear(torch.cat([scores, pop_u.unsqueeze(1), pop_i.unsqueeze(1)], axis=1)).squeeze())
        return torch.stack(rating)

    def score_pairwise_ltr(self, users_emb, items_emb, users, items, pos_or_neg):
        return self.linear(torch.cat([
            self.get_scores(users_emb, items_emb, users, items),
            self.popularity_users[users].unsqueeze(1),
            self.popularity_items[items].unsqueeze(1)
        ], axis=1))
