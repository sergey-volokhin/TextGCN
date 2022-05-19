import torch
from torch import nn

from base_model import BaseModel
from kg_models import DatasetKG
from reviews_models import DatasetReviews


class LTRDataset(DatasetKG, DatasetReviews):

    def __init__(self, args):
        super().__init__(args)
        self._get_users_as_avg_reviews()
        self._get_users_as_avg_descr()

    def _get_users_as_avg_reviews(self):
        ''' use average of reviews to represent users '''
        user_text_embs = {}
        for user, group in self.top_med_reviews.groupby('user_id')['vector']:
            user_text_embs[user] = torch.tensor(group.values.tolist()).mean(axis=0)
        self.users_as_avg_reviews = torch.stack(self.user_mapping['remap_id'].map(
            user_text_embs).values.tolist()).to(self.device)

    def _get_users_as_avg_descr(self):
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

        if not args.load:
            self.logger.error('you need to load pretrained LightGCN model')
            exit()

        if args.freeze_embeddings:
            self.embedding_item.requires_grad_ = False
            self.embedding_user.requires_grad_ = False

        self._setup_representations(args.pos, args.neg)
        self._setup_layers()
        self.score = self.score_ltr
        self.rank_items_for_batch = self.rank_items_for_batch_ltr
        self.evaluate = self.evaluate_ltr

    def _add_vars(self, args):
        super()._add_vars(args)
        self.dense_features = ['gnn', 'reviews', 'desc', 'rev_desc', 'desc_rev', 'gnn_rev', 'gnn_desc']
        self.num_features = 7

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

    def _setup_layers(self):
        ''' dense layer that combines all the scores from different node representations '''
        self.linear = nn.Linear(self.num_features, 1).to(self.device)

    def calculator(self, u, i):
        return torch.sum(torch.mul(u, i), dim=1).unsqueeze(1)

    def get_scores(self, users_emb, items_emb, users, items):
        users_reviews = self.get_user_reviews_mean(users)
        items_reviews = self.get_item_reviews_mean(items)
        users_desc = self.get_user_desc(users)
        items_desc = self.get_item_desc(items)

        user_gnn_rev = torch.cat([users_emb, users_reviews], axis=1)
        item_gnn_rev = torch.cat([items_emb, items_reviews], axis=1)

        user_gnn_desc = torch.cat([users_emb, users_desc], axis=1)
        item_gnn_desc = torch.cat([items_emb, items_desc], axis=1)

        gnn_score = self.calculator(users_emb, items_emb)
        reviews_score = self.calculator(users_reviews, items_reviews)
        desc_score = self.calculator(users_desc, items_desc)
        rev_desc_score = self.calculator(users_reviews, items_desc)
        desc_rev_score = self.calculator(users_desc, items_reviews)
        gnn_rev_score = self.calculator(user_gnn_rev, item_gnn_rev)
        gnn_desc_score = self.calculator(user_gnn_desc, item_gnn_desc)

        return torch.cat([gnn_score,
                          reviews_score,
                          desc_score,
                          rev_desc_score,
                          desc_rev_score,
                          gnn_rev_score,
                          gnn_desc_score], axis=1)

    def score_ltr(self, users_emb, items_emb, users, items, pos_or_neg):
        return self.linear(self.get_scores(users_emb, items_emb, users, items))

    def rank_items_for_batch_ltr(self, users_emb, items_emb, users):
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
        self.num_features += 2
        self.dense_features += ['pop_u', 'pop_i']

    def rank_items_for_batch_ltr(self, users_emb, items_emb, users):
        rating = []
        for user, user_emb in zip(users, users_emb):
            user_emb = user_emb.expand((items_emb.shape[0], user_emb.shape[0]))
            user = torch.tensor(user).expand((items_emb.shape[0]))
            scores = self.get_scores(user_emb, items_emb, user, list(range(self.n_items)))
            pop_u = self.popularity_users[user].expand((items_emb.shape[0],))
            pop_i = self.popularity_items
            rating.append(self.linear(torch.cat([scores, pop_u.unsqueeze(1), pop_i.unsqueeze(1)], axis=1)).squeeze())
        return torch.stack(rating)

    def score_ltr(self, users_emb, items_emb, users, items, pos_or_neg):
        scores = self.get_scores(users_emb, items_emb, users, items)
        pop_u = self.popularity_users[users]
        pop_i = self.popularity_items[items]
        return self.linear(torch.cat([scores, pop_u.unsqueeze(1), pop_i.unsqueeze(1)], axis=1))


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


class LTRSimple(BaseModel):
    ''' concatenate textual vector to item/user representation, only during inference '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        if not args.load:
            self.logger.error('you need to load pretrained LightGCN model')
            exit()
        self.text_user_repr = self.users_as_avg_reviews

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.reviews_df = dataset.reviews_df
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def rank_items_for_batch_ltr(self, users_emb, items_emb, users):
        items_emb = torch.cat([items_emb, self.items_repr[range(self.n_items)]], axis=1)
        batch_users_emb = torch.cat([users_emb, self.text_user_repr[users]], axis=1)
        return torch.matmul(batch_users_emb, items_emb.t())

    def fit(self, batches):
        self.rank_items_for_batch = self.rank_items_for_batch_ltr

        self.logger.info('Performance when using pos=avg:')
        self.items_repr = self.items_as_avg_reviews
        self.evaluate()

        self.logger.info('Performance when using pos=kg:')
        self.items_repr = self.items_as_desc
        self.evaluate()
