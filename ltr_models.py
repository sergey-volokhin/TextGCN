import torch
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from torch import nn

from base_model import BaseModel
from kg_models import DatasetKG
from reviews_models import DatasetReviews


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
        self._setup_layers(args)
        self.score_pairwise = self.score_pairwise_ltr
        self.score_batchwise = self.score_batchwise_ltr

    def _add_vars(self, args):
        super()._add_vars(args)
        self.dense_features = ['gnn', 'reviews', 'desc', 'rev_desc', 'desc_rev', 'gnn_rev', 'gnn_desc']
        if args.freeze:
            self.embedding_user.requires_grad_(False)
            self.embedding_item.requires_grad_(False)

    def _setup_layers(self, args):
        ''' dense layer that combines all the scores from different node representations '''
        args.ltr_layers = [len(self.dense_features)] + args.ltr_layers + [1]
        layers = []
        for i, j in zip(args.ltr_layers, args.ltr_layers[1:]):
            layers.append(nn.Linear(i, j))
        self.layers = nn.Sequential(*layers).to(self.device)

    def get_vectors(self, users_emb, items_emb, users, items):
        vectors = {'users_emb': users_emb, 'items_emb': items_emb}
        vectors['users_reviews'] = self.get_user_reviews_mean(users)
        vectors['items_reviews'] = self.get_item_reviews_mean(items)
        vectors['users_desc'] = self.get_user_desc(users)
        vectors['items_desc'] = self.get_item_desc(items)
        vectors['user_gnn_rev'] = torch.cat([users_emb, vectors['users_reviews']], axis=1)
        vectors['item_gnn_rev'] = torch.cat([items_emb, vectors['items_reviews']], axis=1)
        vectors['user_gnn_desc'] = torch.cat([users_emb, vectors['users_desc']], axis=1)
        vectors['item_gnn_desc'] = torch.cat([items_emb, vectors['items_desc']], axis=1)
        return vectors

    # todo get_scores_batchwise and get_scores_pairwise return scores that differ by 1e-5. why?
    def get_scores_batchwise(self, vectors):
        return torch.cat([
            (vectors['users_emb'] @ vectors['items_emb'].T).unsqueeze(-1),          # gnn          - gnn
            (vectors['users_reviews'] @ vectors['items_reviews'].T).unsqueeze(-1),  # reviews      - reviews
            (vectors['users_desc'] @ vectors['items_desc'].T).unsqueeze(-1),        # desc         - desc
            (vectors['users_reviews'] @ vectors['items_desc'].T).unsqueeze(-1),     # reviews      - desc
            (vectors['users_desc'] @ vectors['items_reviews'].T).unsqueeze(-1),     # desc         - reviews
            (vectors['user_gnn_rev'] @ vectors['item_gnn_rev'].T).unsqueeze(-1),    # gnn||reviews - gnn||reviews
            (vectors['user_gnn_desc'] @ vectors['item_gnn_desc'].T).unsqueeze(-1),  # gnn||desc    - gnn||desc
        ], axis=-1)

    def get_scores_pairwise(self, vectors):

        def sum_mul(x, y):  # todo simplify somehow?
            return torch.sum(torch.mul(x, y), dim=1).unsqueeze(1)

        return torch.cat([
            sum_mul(vectors['users_emb'], vectors['items_emb']),          # gnn          - gnn
            sum_mul(vectors['users_reviews'], vectors['items_reviews']),  # reviews      - reviews
            sum_mul(vectors['users_desc'], vectors['items_desc']),        # desc         - desc
            sum_mul(vectors['users_reviews'], vectors['items_desc']),     # reviews      - desc
            sum_mul(vectors['users_desc'], vectors['items_reviews']),     # desc         - reviews
            sum_mul(vectors['user_gnn_rev'], vectors['item_gnn_rev']),    # gnn||reviews - gnn||reviews
            sum_mul(vectors['user_gnn_desc'], vectors['item_gnn_desc']),  # gnn||desc    - gnn||desc
        ], axis=1)

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        vectors = self.get_vectors(users_emb, items_emb, users, list(range(self.n_items)))
        return self.layers(self.get_scores_batchwise(vectors)).squeeze()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        vectors = self.get_vectors(users_emb, items_emb, users, items)
        return self.layers(self.get_scores_pairwise(vectors))


class LTRLinearWPop(LTRLinear):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.popularity_users = dataset.popularity_users
        self.popularity_items = dataset.popularity_items

    def _add_vars(self, args):
        super()._add_vars(args)
        self.dense_features += ['pop_u', 'pop_i']

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        vectors = self.get_vectors(users_emb, items_emb, users, list(range(self.n_items)))
        features = self.get_scores_batchwise(vectors)

        pop_u = self.popularity_users[users].unsqueeze(1)
        pop_u = pop_u.unsqueeze(-1).expand(pop_u.shape[0], self.n_items, pop_u.shape[-1])
        pop_i = self.popularity_items
        pop_i = pop_i.unsqueeze(1).expand(len(users), pop_i.shape[0], 1)

        cat = torch.cat([features, pop_u, pop_i], axis=-1)
        return self.layers(cat).squeeze()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        vectors = self.get_vectors(users_emb, items_emb, users, items)
        features = self.get_scores_pairwise(vectors)
        pop_u = self.popularity_users[users].unsqueeze(1)
        pop_i = self.popularity_items[items].unsqueeze(1)
        cat = torch.cat([features, pop_u, pop_i], axis=-1)
        return self.layers(cat)
