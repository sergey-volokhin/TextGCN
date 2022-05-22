import torch
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from torch import nn
from tqdm.auto import tqdm

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
        if args.freeze_embeddings:
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
        users_reviews = self.get_user_reviews_mean(users)
        items_reviews = self.get_item_reviews_mean(items)
        users_desc = self.get_user_desc(users)
        items_desc = self.get_item_desc(items)
        user_gnn_rev = torch.cat([users_emb, users_reviews], axis=1)
        item_gnn_rev = torch.cat([items_emb, items_reviews], axis=1)
        user_gnn_desc = torch.cat([users_emb, users_desc], axis=1)
        item_gnn_desc = torch.cat([items_emb, items_desc], axis=1)
        return [users_reviews,
                items_reviews,
                user_gnn_rev,
                items_desc,
                users_desc,
                item_gnn_rev,
                user_gnn_desc,
                item_gnn_desc]

    # todo get_scores_batchwise and get_scores_pairwise return scores that differ by 1e-5. why?
    def get_scores_batchwise(self,
                             users_emb,
                             items_emb,
                             users_reviews,
                             items_reviews,
                             user_gnn_rev,
                             items_desc,
                             users_desc,
                             item_gnn_rev,
                             user_gnn_desc,
                             item_gnn_desc):
        return torch.cat([
            (users_emb @ items_emb.T).unsqueeze(-1),          # gnn          - gnn
            (users_reviews @ items_reviews.T).unsqueeze(-1),  # reviews      - reviews
            (users_desc @ items_desc.T).unsqueeze(-1),        # desc         - desc
            (users_reviews @ items_desc.T).unsqueeze(-1),     # reviews      - desc
            (users_desc @ items_reviews.T).unsqueeze(-1),     # desc         - reviews
            (user_gnn_rev @ item_gnn_rev.T).unsqueeze(-1),    # gnn||reviews - gnn||reviews
            (user_gnn_desc @ item_gnn_desc.T).unsqueeze(-1),  # gnn||desc    - gnn||desc
        ], axis=-1)

    def get_scores_pairwise(self,
                            users_emb,
                            items_emb,
                            users_reviews,
                            items_reviews,
                            user_gnn_rev,
                            items_desc,
                            users_desc,
                            item_gnn_rev,
                            user_gnn_desc,
                            item_gnn_desc):
        return torch.cat([
            torch.sum(torch.mul(users_emb, items_emb), dim=1).unsqueeze(1),          # gnn          - gnn
            torch.sum(torch.mul(users_reviews, items_reviews), dim=1).unsqueeze(1),  # reviews      - reviews
            torch.sum(torch.mul(users_desc, items_desc), dim=1).unsqueeze(1),        # desc         - desc
            torch.sum(torch.mul(users_reviews, items_desc), dim=1).unsqueeze(1),     # reviews      - desc
            torch.sum(torch.mul(users_desc, items_reviews), dim=1).unsqueeze(1),     # desc         - reviews
            torch.sum(torch.mul(user_gnn_rev, item_gnn_rev), dim=1).unsqueeze(1),    # gnn||reviews - gnn||reviews
            torch.sum(torch.mul(user_gnn_desc, item_gnn_desc), dim=1).unsqueeze(1),  # gnn||desc    - gnn||desc
        ], axis=1)

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        vectors = self.get_vectors(users_emb, items_emb, users, list(range(self.n_items)))
        return self.layers(self.get_scores_batchwise(users_emb, items_emb, *vectors)).squeeze()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        vectors = self.get_vectors(users_emb, items_emb, users, items)
        return self.layers(self.get_scores_pairwise(users_emb, items_emb, *vectors))


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
        features = self.get_scores_batchwise(users_emb, items_emb, *vectors)

        pop_u = self.popularity_users[users].unsqueeze(1)
        pop_u = pop_u.expand(self.n_items, pop_u.shape[0], pop_u.shape[1]).T.unsqueeze(-1).squeeze(0)

        pop_i = self.popularity_items
        pop_i = pop_i.unsqueeze(1).expand(len(users), pop_i.shape[0], 1)

        cat = torch.cat([features, pop_u, pop_i], axis=-1)
        return self.layers(cat).squeeze()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        vectors = self.get_vectors(users_emb, items_emb, users, items)
        features = self.get_scores_pairwise(users_emb, items_emb, *vectors)
        pop_u = self.popularity_users[users].unsqueeze(1)
        pop_i = self.popularity_items[items].unsqueeze(1)
        cat = torch.cat([features, pop_u, pop_i], axis=-1)
        return self.layers(cat)
