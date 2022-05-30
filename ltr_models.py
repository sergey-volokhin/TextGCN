import torch
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from torch import nn
from tqdm.auto import tqdm
from xgboost import XGBRanker


from base_model import BaseModel
from kg_models import DatasetKG
from reviews_models import DatasetReviews
from utils import early_stop


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
        base class for Learning-to-Rank models, which uses pre-trained
        LightGCN vectors and trains a layer on top
    '''

    def _copy_args(self, args):
        super()._copy_args(args)
        self.load_base = args.load_base
        self.freeze = args.freeze

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.users_as_avg_desc = dataset.users_as_avg_desc
        self.items_as_desc = dataset.items_as_desc

    def _init_embeddings(self, emb_size):
        super()._init_embeddings(emb_size)
        if self.freeze:
            self.embedding_user.requires_grad_(False)
            self.embedding_item.requires_grad_(False)

    def _add_vars(self, args):
        super()._add_vars(args)

        ''' load the base model, before overwriting the scoring functions '''
        if self.load_base:
            self.load_model(self.load_base)

        ''' features we are going to use'''
        self.feature_names = ['gnn',
                              'reviews',
                              'desc',
                              'reviews-description',
                              'description-reviews',
                              'gnn-reviews',
                              'gnn-description']
        ''' build the trainable layer on top '''
        self._setup_layers(args)

        ''' overwrite the scoring methods '''
        self.score_pairwise = self.score_pairwise_ltr
        self.score_batchwise = self.score_batchwise_ltr

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

    def get_item_reviews_mean(self, i, u=None):
        ''' represent items as mean of their reviews '''
        return self.items_as_avg_reviews[i]

    def get_item_desc(self, i, u=None):
        ''' represent items as their description '''
        return self.items_as_desc[i]

    def get_item_reviews_user(self, i, u=None):
        ''' represent items as the review of corresponding user '''
        df = self.reviews_df.loc[torch.stack([i, u], axis=1).tolist()]
        return torch.tensor(df.values.tolist()).to(self.device)

    def get_vectors(self, users_emb, items_emb, users, items):
        '''
            returns all the dense features that consists of textual representations:
                user/item LightGCN vectors
                user/item reviews
                user/item descriptions
                & combinations of aforementioned
        '''
        vectors = {'users_emb': users_emb, 'items_emb': items_emb}
        vectors['users_desc'] = self.get_user_desc(users)
        vectors['items_desc'] = self.get_item_desc(items)
        vectors['users_reviews'] = self.get_user_reviews_mean(users)
        vectors['items_reviews'] = self.get_item_reviews_mean(items)
        vectors['user_gnn_desc'] = torch.cat([users_emb, vectors['users_desc']], axis=-1)
        vectors['item_gnn_desc'] = torch.cat([items_emb, vectors['items_desc']], axis=-1)
        vectors['user_gnn_reviews'] = torch.cat([users_emb, vectors['users_reviews']], axis=-1)
        vectors['item_gnn_reviews'] = torch.cat([items_emb, vectors['items_reviews']], axis=-1)
        return vectors

    # todo get_scores_batchwise and get_scores_pairwise return scores that differ by 1e-5. why?
    def get_features_batchwise(self, vectors):
        '''
            batchwise (all-to-all) calculation of features for top layer:
            vectors['users_emb'].shape = (batch_size, emb_size)
            vectors['items_emb'].shape = (n_items, emb_size)
        '''
        return torch.cat([
            (vectors['users_emb'] @ vectors['items_emb'].T).unsqueeze(-1),
            (vectors['users_reviews'] @ vectors['items_reviews'].T).unsqueeze(-1),
            (vectors['users_desc'] @ vectors['items_desc'].T).unsqueeze(-1),
            (vectors['users_reviews'] @ vectors['items_desc'].T).unsqueeze(-1),
            (vectors['users_desc'] @ vectors['items_reviews'].T).unsqueeze(-1),
            (vectors['user_gnn_reviews'] @ vectors['item_gnn_reviews'].T).unsqueeze(-1),
            (vectors['user_gnn_desc'] @ vectors['item_gnn_desc'].T).unsqueeze(-1),
        ], axis=-1)

    def get_features_pairwise(self, vectors):
        '''
            pairwise (1-to-1) calculation of features for top layer:
            vectors['users_emb'].shape == vectors['items_emb'].shape
        '''

        def sum_mul(x, y):
            return (x * y).sum(dim=1).unsqueeze(1)

        return torch.cat([
            sum_mul(vectors['users_emb'], vectors['items_emb']),
            sum_mul(vectors['users_reviews'], vectors['items_reviews']),
            sum_mul(vectors['users_desc'], vectors['items_desc']),
            sum_mul(vectors['users_reviews'], vectors['items_desc']),
            sum_mul(vectors['users_desc'], vectors['items_reviews']),
            sum_mul(vectors['user_gnn_reviews'], vectors['item_gnn_reviews']),
            sum_mul(vectors['user_gnn_desc'], vectors['item_gnn_desc']),
        ], axis=1)


class LTRLinear(LTRBase):
    ''' trains a dense layer on top of LightGCN '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.evaluate = self.evaluate_ltr

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

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        vectors = self.get_vectors(users_emb, items_emb, users, list(range(self.n_items)))
        return self.layers(self.get_features_batchwise(vectors)).squeeze()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        vectors = self.get_vectors(users_emb, items_emb, users, items)
        return self.layers(self.get_features_pairwise(vectors))

    def evaluate_ltr(self):
        ''' print weights (i.e. feature importances) if the model consists of single layer '''
        res = super().evaluate()
        if len(self.layers) == 1:
            for f, w in zip(self.feature_names, self.layers[0].weight.tolist()[0]):
                self.logger.info(f'{f:<20} {w:.4}')
        return res


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
        vectors = self.get_vectors(users_emb, items_emb, users, list(range(self.n_items)))
        features = self.get_features_batchwise(vectors)
        pop_u = self.popularity_users[users].unsqueeze(-1).expand(len(users), self.n_items, 1)
        pop_i = self.popularity_items.expand(len(users), self.n_items, 1)
        return self.layers(torch.cat([features, pop_u, pop_i], axis=-1)).squeeze()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        vectors = self.get_vectors(users_emb, items_emb, users, items)
        features = self.get_features_pairwise(vectors)
        return self.layers(torch.cat([features,
                                      self.popularity_users[users],
                                      self.popularity_items[items]], axis=-1))


class LTRGBDT(LTRBase):
    ''' train a Gradient Boosted Decision Tree (sklearn) on top of LightGCN '''

    def _setup_layers(self, args):
        self.tree = GBRT(n_estimators=10, max_depth=3)

    def fit(self, batches):

        vectors_pos, vectors_neg = [], []
        users_emb, items_emb = self.representation
        for data in tqdm(batches,
                         desc='batches',
                         leave=False,
                         dynamic_ncols=True,
                         disable=self.slurm):

            data = data.t()
            users, pos, negs = data[0], data[1], data[2:]
            users_emb = users_emb[users]
            pos_features = self.score_pairwise(users_emb, items_emb[pos], users, pos)
            vectors_pos.append(pos_features)
            for neg in negs:
                neg_features = self.score_pairwise(users_emb, items_emb[neg], users, neg)
                vectors_neg.append(neg_features)

        vectors_pos = torch.stack(vectors_pos, dim=1)
        vectors_neg = torch.stack(vectors_neg, dim=1)

        features = torch.cat([vectors_pos, vectors_neg]).squeeze()
        y_true = torch.cat([torch.ones(features.shape[0] // 2), torch.zeros(features.shape[0] // 2)])

        self.tree.fit(features.cpu().detach().numpy(), y_true.cpu().detach().numpy())
        self.evaluate()
        self.checkpoint()

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        vectors = self.get_vectors(users_emb, items_emb, users, items)
        return self.get_scores_pairwise(users_emb, items_emb, *vectors)

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        vectors = self.get_vectors(users_emb, items_emb, users, list(range(self.n_items)))
        x = self.get_scores_batchwise(users_emb, items_emb, *vectors).detach()
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[-1]).cpu().numpy()
        return self.tree.predict(x)


class LTRXGBoost(LTRBase):

    def _setup_layers(self, args):
        self.tree = XGBRanker(verbosity=1)  # n_estimators, max_depth, verbosity=0-3
        # objective = 'rank:pairwise', 'rank:ndcg', 'rank:map'
        # tree_method = 'gpu_hist', 'exact'
        # booster = 'gbtree', 'gblinear', 'dart'
        # max_bin
        # n_jobs
        # sampling_method = 'gradient_based'
        # predictor = 'gpu_predictor'
        # eval_metric = ['f1', 'recall', 'precision', 'ndcg', ]

    def fit(self, batches):
        users_emb, items_emb = self.representation
        for data in tqdm(batches,
                         desc='batches',
                         leave=False,
                         dynamic_ncols=True,
                         disable=self.slurm):

            data = data.t()
            users, pos, negs = data[0], data[1], data[2:]
            vectors = self.get_vectors(users_emb[users], items_emb[pos], users, pos)
            y_true = []
            groups = []
            features = []
            features.append(self.get_features_pairwise(vectors))

            for neg in negs:
                vectors = self.get_vectors(users_emb[users], items_emb[neg], users, neg)
                neg_features = self.get_features_pairwise(vectors)
                features.append(neg_features)

            y_true += [[1] + [0] * len(negs)] * len(users)
            groups.append([len(negs) + 1] * len(users))

            features = torch.stack(features).reshape(-1, len(self.feature_names))

            self.tree.fit(features.cpu().detach().numpy(), y_true, group=groups)

        self.evaluate()
        self.checkpoint(-1)

    # def score_pairwise_ltr(self, users_emb, items_emb, users, items):
    #     vectors = self.get_vectors(users_emb, items_emb, users, items)
    #     return self.layers(self.get_features_pairwise(vectors))


# df containing rows with feature columns,
# a column named id which identifies the group,
# and a target column called rank which contains the target.
