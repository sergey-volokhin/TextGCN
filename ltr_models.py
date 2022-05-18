import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from base_model import BaseModel
from kg_models import DatasetKG
from reviews_models import DatasetReviews


class LTRDataset(DatasetKG, DatasetReviews):

    def __init__(self, args):
        super().__init__(args)
        self._get_users_as_avg_reviews()

    def _get_users_as_avg_reviews(self):
        ''' use average of reviews to represent users '''
        user_text_embs = {}
        for user, group in self.top_med_reviews.groupby('user_id')['vector']:
            user_text_embs[user] = torch.tensor(group.values.tolist()).mean(axis=0)
        self.users_as_avg_reviews = torch.stack(self.user_mapping['remap_id'].map(
            user_text_embs).values.tolist()).to(self.device)


class LTRBase(BaseModel):
    '''
        base class for LTR models, which uses pre-trained
        LightGCN vectors to train a layer on top
    '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        if not self.load:
            self.logger.error('you need to load pretrained LightGCN model')
            exit()

        self.logger.info('Initial model metrics:')
        self.evaluate()
        self.metrics_logger = {i: np.zeros((0, len(self.k))) for i in self.metrics}

    def _copy_args(self, args):
        super()._copy_args(args)
        self.load = args.load

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.reviews_df = dataset.reviews_df
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def _add_vars(self, args):
        super()._add_vars(args)

        ''' set up the representation functions for items and users'''
        if args.pos == 'avg':
            self.items_repr = self.items_as_avg_reviews
        elif args.pos == 'kg':
            self.items_repr = self.items_as_desc

        self.text_user_repr = self.users_as_avg_reviews

    def rank_items_for_batch_ltr(self):
        ''' how do we predict the score using text data? '''
        raise NotImplementedError

    ''' utilities: representation functions '''

    def get_user_reviews_mean(self, u):
        ''' represent users with mean of their reviews '''
        return self.users_as_avg_reviews[u]

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


class LTRSimple(LTRBase):
    ''' concatenate textual vector to item/user representation, only during inference '''

    def rank_items_for_batch_ltr(self, users_emb, items_emb, users):
        items_emb = torch.cat([items_emb, self.items_repr[range(self.n_items)]], axis=1)
        batch_users_emb = torch.cat([users_emb, self.text_user_repr[users]], axis=1)
        return torch.matmul(batch_users_emb, items_emb.t())

    def fit(self, batches):
        self.rank_items_for_batch = self.rank_items_for_batch_ltr
        self.evaluate()


class LTRLinear(LTRBase):
    ''' trains a dense layer on top of GNN '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self._setup_layers(args.emb_size)
        self._setup_representations(args.pos, args.neg)
        self.score = self.score_ltr
        self.rank_items_for_batch = self.rank_items_for_batch_ltr

    def _setup_representations(self, pos, neg):
        self.get_pos_items_reprs = {'avg': self.get_item_reviews_mean,
                                    'kg': self.get_item_desc,
                                    'user': self.get_item_reviews_user
                                    }[pos]
        self.get_neg_items_reprs = {'avg': self.get_item_reviews_mean,
                                    'kg': self.get_item_desc
                                    }[neg]
        ''' currently the only one way to represent users is with mean of their reviews '''
        self.get_text_users_reprs = self.get_user_reviews_mean

    def _setup_layers(self, emb_size):
        self.distill_text = nn.Linear(384, emb_size).to(self.device)  # TODO remove magic dim of embeddings
        num_features = emb_size * 4 + 1
        self.linear = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Linear(num_features, 1),
        ).to(self.device)

    def get_node_repr(self, emb, text):
        ''' returns the concatenated vector gnn||text for item/user '''
        return torch.cat([emb, F.normalize(self.distill_text(text), dim=0)], axis=1)

    def score_ltr(self, users_emb, items_emb, users, items, pos_or_neg):
        user_vectors = self.get_node_repr(users_emb, self.get_text_users_reprs(users))
        fn = self.get_pos_items_reprs if pos_or_neg == 'pos' else self.get_neg_items_reprs
        item_vectors = self.get_node_repr(items_emb, fn(items))
        scores = torch.sum(torch.mul(user_vectors, item_vectors), dim=1).unsqueeze(1)
        return self.linear(torch.cat([user_vectors, item_vectors, scores], axis=1))

    def rank_items_for_batch_ltr(self, users_emb, items_emb, users):
        # !right now we represent items for evaluation as negatives
        item_vectors = self.get_node_repr(items_emb, self.get_neg_items_reprs(range(self.n_items)))
        user_vectors = self.get_node_repr(users_emb, self.get_text_users_reprs(users))
        scores = torch.matmul(user_vectors, item_vectors.t())
        rating = []
        for user, score in zip(user_vectors, scores):
            user_repeated = user.expand((item_vectors.shape[0], user.shape[0]))
            rating.append(self.linear(torch.cat([user_repeated, item_vectors, score.unsqueeze(1)], axis=1)).squeeze())
            # rating.append(self.linear(torch.cat([score.unsqueeze(1)], axis=1)).squeeze())
        return torch.stack(rating)


class LTRLinearFeatures(LTRLinear):
    '''
        trains a dense layer on top of GNN,
        uses hand-crafted features in addition to raw text and gnn vectors
    '''

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.popularity_users = dataset.popularity_users
        self.popularity_items = dataset.popularity_items

    def _setup_layers(self, emb_size):
        self.distill_text = nn.Linear(384, emb_size).to(self.device)
        num_features = emb_size * 4 + 3  # 2 popularities and cosine score
        self.linear = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Linear(num_features, 1),
        ).to(self.device)

    def get_node_repr(self, emb, text, popularity):
        ''' returns the concatenated vector gnn||text||popularity for item/user '''
        distil = F.normalize(self.distill_text(text), dim=0)
        return torch.cat([emb, distil, popularity.unsqueeze(1)], axis=1)

    def score_ltr(self, users_emb, items_emb, users, items, pos_or_neg):
        user_vectors = self.get_node_repr(users_emb, self.get_text_users_reprs(users), self.popularity_users[users])
        fn = self.get_pos_items_reprs if pos_or_neg == 'pos' else self.get_neg_items_reprs
        item_vectors = self.get_node_repr(items_emb, fn(items), self.popularity_items[items])
        scores = torch.sum(torch.mul(users_emb, items_emb), dim=1).unsqueeze(1)
        return self.linear(torch.cat([user_vectors, item_vectors, scores], axis=1))

    def rank_items_for_batch_ltr(self, users_emb, items_emb, users):
        item_vectors = self.get_node_repr(items_emb, self.get_neg_items_reprs(range(self.n_items)), self.popularity_items)
        user_vectors = self.get_node_repr(users_emb, self.get_text_users_reprs(users), self.popularity_users[users])
        # scores = (user_vectors.unsqueeze(1) * item_vectors).sum(axis=-1)
        scores = torch.matmul(user_vectors, item_vectors.t())
        rating = []
        for user, cur_scores in zip(user_vectors, scores):
            user_repeated = user.expand((item_vectors.shape[0], user.shape[0]))
            vectors = [user_repeated, item_vectors, cur_scores.unsqueeze(1)]
            rating.append(self.linear(torch.cat(vectors, axis=1)).squeeze())
        return torch.stack(rating)


class LTRCosine(BaseModel):
    ''' train the LightGCN model from scratch, concatenating text to GNN vectors '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        if args.model == 'ltr_reviews':
            self.items_repr = self.items_as_avg_reviews
        elif args.model == 'ltr_kg':
            self.items_repr = self.items_as_desc
        else:
            raise AttributeError(f'wrong LTR model name: "{args.model}"')

        self.text_user_repr = self.users_as_avg_reviews

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def score(self, users_emb, item_emb, users, items, pos_or_neg=None):
        user_part = torch.cat([users_emb, self.users_as_avg_reviews[users]], axis=1)
        item_part = torch.cat([item_emb, self.items_repr[items]], axis=1)
        return torch.sum(torch.mul(user_part, item_part), dim=1)

    def rank_items_for_batch(self, users_emb, items_emb, users):
        items_emb = torch.cat([items_emb, self.items_repr[range(self.n_items)]], axis=1)
        batch_users_emb = torch.cat([users_emb, self.text_user_repr[users]], axis=1)
        return torch.matmul(batch_users_emb, items_emb.t())
