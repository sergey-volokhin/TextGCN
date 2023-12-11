import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier as GBCT
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from tqdm.auto import tqdm, trange
from xgboost import XGBRanker

from .ltr_models import LTRBase, LTRDataset


class OneBatchDataset(LTRDataset):
    ''' returns the zeros tensor with 1s in 'positives' places for that user'''

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        y_true = torch.zeros(self.n_items)
        y_true[self.positive_lists[idx]['tensor']] = 1
        return idx, y_true


class LTRGradientBoosted(LTRBase):
    '''
        fit an sklearn-type 1-epoch-fit model on top of LightGCN
        iteratively adds batches of data to fit *additional* trees.
        uses *ALL* unobserved entries as negatives
    '''

    def _setup_layers(self, config):
        self.type = config.model
        if 'xgboost' in config.model:
            self.tree = XGBRanker(
                objective='rank:ndcg',
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                gpu_id=1,

                # n_estimators=config.n_estimators,
                # max_depth=config.max_depth,
                # min_child_weight=config.min_child_weight,
                # eta=config.ltr_eta,
                # n_estimators=10,
                n_estimators=75,
                min_child_weight=15,
                eta=0.6,

                eval_metric=['auc', 'ndcg@20', 'aucpr', 'map@20'],
            )
            self.tree_fit = self.xgboost_fit
        elif 'gbdt' in config.model:
            self.tree = GBRT(
                n_estimators=10,
                max_depth=3,
                warm_start=True,
                verbose=True,
            )
            self.tree_fit = self.gbdt_fit

    def xgboost_fit(self, features, y_true, groups):
        if self.warm:
            self.tree.fit(features, y_true, group=groups, xgb_model=self.tree)
        else:
            self.tree.fit(features, y_true, group=groups)

    def gbdt_fit(self, features, y_true, groups):
        self.tree.fit(features, y_true)

    def fit(self, batches):
        self.training = True
        users_emb, items_emb = self.forward()
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        self.warm = False
        for data in tqdm(batches,
                         desc='train batches',
                         leave=False,
                         dynamic_ncols=True,
                         disable=self.slurm):
            users, y_true = data[0], data[1].flatten()
            u_vecs = self.get_user_vectors(users_emb[users], users)
            features = self.get_features_batchwise(u_vecs, i_vecs).squeeze()
            features = features.reshape((-1, len(self.feature_names)))
            groups = [self.n_items] * len(users)  # each query has all items
            self.tree_fit(features.detach().cpu().numpy(), y_true, groups)
            self.warm = True

        self.evaluate()
        self.checkpoint(-1)
        self.logger.info(list(zip(self.feature_names, self.tree.feature_importances_)))

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        features = self.get_features_batchwise(u_vecs, i_vecs).detach().cpu()
        results = self.tree.predict(features.reshape(-1, features.shape[-1]))
        return torch.from_numpy(results.reshape(features.shape[:2]))


class LTRGradientBoostedWPop(LTRGradientBoosted):
    ''' extends LTRLinear by adding popularity features '''

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.popularity_users = dataset.popularity_users
        self.popularity_items = dataset.popularity_items

    def _setup_layers(self, config):
        self.feature_names += ['user popularity', 'item popularity']
        super()._setup_layers(config)

    def fit(self, batches):
        self.training = True
        users_emb, items_emb = self.forward()
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        self.warm = False
        for data in tqdm(batches,
                         desc='train batches',
                         leave=False,
                         dynamic_ncols=True,
                         disable=self.slurm):
            users, y_true = data[0], data[1].flatten()
            u_vecs = self.get_user_vectors(users_emb[users], users)
            features = self.get_features_batchwise(u_vecs, i_vecs).squeeze()
            pop_u = self.popularity_users[users].unsqueeze(-1).expand(len(users), self.n_items, 1)
            pop_i = self.popularity_items.expand(len(users), self.n_items, 1)
            features = torch.cat([features, pop_u, pop_i], axis=-1)
            features = features.reshape((-1, len(self.feature_names)))
            groups = [self.n_items] * len(users)  # each query has all items
            self.tree_fit(features.detach().cpu().numpy(), y_true, groups)
            self.warm = True

        self.evaluate()
        self.checkpoint(-1)
        self.logger.info(list(zip(self.feature_names, self.tree.feature_importances_)))

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        features = self.get_features_batchwise(u_vecs, i_vecs)
        pop_u = self.popularity_users[users].unsqueeze(-1).expand(len(users), self.n_items, 1)
        pop_i = self.popularity_items.expand(len(users), self.n_items, 1)
        features = torch.cat([features, pop_u, pop_i], axis=-1)
        results = self.tree.predict(features.reshape(-1, features.shape[-1]).detach().cpu())
        return torch.from_numpy(results.reshape(features.shape[:2]))

    def score_pairwise_ltr(self, users_emb, items_emb, users, items):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, items)
        features = self.get_features_batchwise(u_vecs, i_vecs).detach().cpu()
        features = torch.cat([features,
                              self.popularity_users[users],
                              self.popularity_items[items]])
        results = self.tree.predict(features.reshape(-1, features.shape[-1]))
        return torch.from_numpy(results.reshape(features.shape[:2]))


class MarcusGradientBoosted(LTRGradientBoosted):

    def _setup_layers(self, config):
        config.model = 'xgboost'
        super()._setup_layers(config)

    def fit(self, batches):
        self.training = True
        users_emb, items_emb = self.forward()
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        self.warm = False
        for epoch in trange(self.epochs, desc='training', leave=False, dynamic_ncols=True):
            for data in tqdm(batches,
                             desc='train batches',
                             leave=False,
                             dynamic_ncols=True,
                             disable=self.slurm):
                users, y_true = data[0], data[1]
                groups, features, labels = [], [], []
                for n in range(len(users)):
                    u, yt = users[n], y_true[n]
                    pos_idx = yt.nonzero().squeeze()
                    neg_idx = (1 - yt).nonzero().squeeze()
                    neg_sample_idx = []
                    for _ in range(self.neg_samples):
                        neg_sample_idx.append(neg_idx[torch.randint(len(neg_idx), (len(pos_idx), ))])
                    neg_sample_idx = torch.stack(neg_sample_idx, axis=1)
                    sample_idx = torch.concat([pos_idx, neg_sample_idx])

                    u_vec = self.get_user_vectors(users_emb[u], u)
                    i_vecs = self.get_item_vectors(items_emb[sample_idx], sample_idx)
                    features.append(self.get_features_batchwise(u_vec, i_vecs).squeeze())
                    labels.append(yt[sample_idx])
                    groups.append(len(sample_idx))

                features = torch.concat(features)
                y_true = torch.concat(labels)

                self.tree_fit(features.detach().cpu().numpy(), y_true, groups)
                self.warm = True

            self.evaluate()
            self.checkpoint(-1)
