import torch
from tqdm import tqdm
from xgboost import XGBRanker
from sklearn.ensemble import GradientBoostingRegressor as GBRT

from ltr_models import LTRBase, LTRDataset


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

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.score_batchwise = self.score_batchwise_ltr

    def _setup_layers(self, args):
        self.type = args.model
        if args.model == 'xgboost':
            self.tree = XGBRanker(
                n_estimators=10,
                objective='rank:ndcg',
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                #   max_depth=4,
                #   min_child_weight=3,
                #   eta=0.1,
                eval_metric=['auc', 'ndcg@20', 'aucpr', 'map@20'],
            )
            self.tree_fit = self.xgboost_fit
        elif args.model == 'gbdt':
            self.tree = GBRT(
                n_estimators=10,
                max_depth=3,
                warm_start=True,
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
        users_emb, items_emb = self.representation
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

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        u_vecs = self.get_user_vectors(users_emb, users)
        i_vecs = self.get_item_vectors(items_emb, self.all_items)
        features = self.get_features_batchwise(u_vecs, i_vecs).detach().cpu()
        results = self.tree.predict(features.reshape(-1, features.shape[-1]))
        return torch.from_numpy(results.reshape(features.shape[:2]))
