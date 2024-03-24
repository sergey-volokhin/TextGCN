import torch
from torch import nn

from .BaseModel import BaseModel
from .DatasetKG import DatasetKG
from .DatasetRanking import DatasetRanking
from .DatasetReviews import DatasetReviews
from .DatasetScoring import DatasetScoring


class LTRDataset(DatasetKG, DatasetReviews):
    ''' combines KG and Reviews datasets '''

    def __init__(self, config):
        super().__init__(config)
        self._get_users_as_avg_desc()

    def _get_users_as_avg_desc(self):
        ''' use mean of kg features of items reviewed to represent users '''
        kg_feat_user_text_embs = {i: {} for i in self.kg_features}
        for user, group in self.top_med_reviews.groupby('user_id')['asin']:
            for feature in self.kg_features:
                kg_feat_user_text_embs[feature][user] = self.item_representations[feature][group.values].mean(axis=0).cpu()

        for feature in self.kg_features:
            self.user_representations[feature] = torch.stack(
                self.user_mapping['remap_id']
                .map(kg_feat_user_text_embs[feature])
                .values
                .tolist()
            ).to(self.device)


class LTRDatasetRank(LTRDataset, DatasetRanking):
    ''' combines KG and Reviews datasets and ranking dataset '''


class LTRDatasetScore(LTRDataset, DatasetScoring):
    ''' combines KG and Reviews datasets with scoring dataset '''


class LTRBaseModel(BaseModel):
    '''
    Objective-agnostic model that joins GCN model
    with textual features (reviews and descriptions)
    trains something on top
    '''

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._setup_layers(config)
        self._load_base(config, dataset)

    def _copy_params(self, config):
        super()._copy_params(config)
        self.features = config.ltr_text_features  # all textual features used (usr_repr-item_repr)
        self.kg_features = config.kg_features

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.user_representations = dataset.user_representations
        self.item_representations = dataset.item_representations
        self.all_items = dataset.all_items

    def _load_base(self, config, dataset):
        ''' load the base model '''
        self.foundation = self.foundation_class(config, dataset)
        if config.load_base:
            self.logger.info(f'Loading base LightGCN model from {config.load_base}.')
            if not config.freeze:
                self.logger.warn('Base model not frozen for LTR model, this will degrade performance')
            self.foundation.load(config.load_base)
            results = self.foundation.evaluate()
            self.metrics_log += results
            self.logger.info('Base model metrics:')
            self.metrics_log.log()
            self.metrics_log -= results
        else:
            self.logger.warn('Not using a pretrained base model leads to poor performance.')

    def _add_vars(self, config):
        super()._add_vars(config)

        '''
        features we are going to use, comprise of user and item representations,
        which are then dot-producted to get the similarity score between those representations
        user_rep and item_rep keys have to be in user_vectors and item_vectors dictionaries, defined in
        `self.get_user_vectors` and `self.get_item_vectors` functions
        '''
        self.features = {i: {'user_rep': i.split('-')[0], 'item_rep': i.split('-')[1]} for i in self.features}
        self.features['base_model_score'] = {'user_rep': 'emb', 'item_rep': 'emb'}
        self.feature_names, self.feature_build = list(zip(*self.features.items()))

    def _setup_layers(self, config):
        '''
        dense layers that combine all the scores from different node representations
        layer_sizes: represents the size and number of hidden layers (default: no hidden layers)
        '''
        layer_sizes = [len(self.feature_names)] + config.ltr_layers + [1]
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

    def get_item_reviews_user(self, i, u):  # not used
        ''' represent items as the reviews of corresponding user '''
        df = self.reviews_vectors.loc[torch.stack([i, u], axis=1).tolist()]
        return torch.tensor(df.values.tolist()).to(self.device)

    def get_item_vectors(self, items_emb, items):
        ''' get vectors used to calculate textual representations dense features for items '''
        return {
            'emb': items_emb,
            'reviews': self.item_representations['reviews'][items],  # items as mean of their reviews
            **{i: self.item_representations[i][items] for i in self.kg_features},  # all other kg features
        }

    def get_user_vectors(self, users_emb, users):
        ''' get vectors used to calculate textual representations dense features for users '''
        return {
            'emb': users_emb,
            'reviews': self.user_representations['reviews'][users],  # users as mean of their reviews
            **{i: self.user_representations[i][users] for i in self.kg_features},  # all other kg features
        }

    # todo get_scores_batchwise and get_scores_pairwise return scores that differ by 1e-5. why?
    def get_features_batchwise(self, u_vecs, i_vecs):
        '''
        batchwise (all-to-all) calculation of features for top layer:
        vectors['users_emb'].shape = (batch_size, emb_size)
        vectors['items_emb'].shape = (n_items, emb_size)
        '''
        return torch.cat([(u_vecs[i['user_rep']] @ i_vecs[i['item_rep']].T).unsqueeze(-1) for i in self.feature_build], axis=-1)

    def get_features_pairwise(self, u_vecs, i_vecs):
        '''
        pairwise (1-to-1) calculation of features for top layer:
        vectors['users_emb'].shape == vectors['items_emb'].shape
        '''

        def sum_mul(x, y):
            return (x * y).sum(dim=1).unsqueeze(1)

        return torch.cat([(sum_mul(u_vecs[i['user_rep']], i_vecs[i['item_rep']])) for i in self.feature_build], axis=1)


class LTRBaseWPop(LTRBaseModel):
    ''' extends LTRLinear by adding popularity features '''
    # popularity feature is not calculated as a similarity
    # hence not including it in feature_build

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.popularity_users = dataset.popularity_users
        self.popularity_items = dataset.popularity_items

    def _add_vars(self, config):
        super()._add_vars(config)
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
