import random

import torch
from tqdm import tqdm, trange
from xgboost import XGBRanker

from ltr_models import LTRBase, LTRDataset


# class XGBoostDataset(LTRDataset, AdvSamplDataset):
#     pass

class XGBoostDataset(LTRDataset):

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return idx // self.bucket_len


class LTRXGBoost(LTRBase):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.positive_lists = dataset.positive_lists
        self.all_items = torch.arange(0, self.n_items).to(self.device)
        self.score_batchwise = self.score_batchwise_ltr

    def _setup_layers(self, args):
        self.tree = XGBRanker(verbosity=1,
                              objective='rank:pairwise',
                              tree_method='gpu_hist',
                              predictor='gpu_predictor',
                              eval_metric=['auc', 'ndcg@20', 'aucpr', 'map@20'],
                              )

        #   eval_metric=['recall', 'precision', 'ndcg'])
        # n_estimators, max_depth, verbosity=0-3
        # objective='rank:pairwise', 'rank:ndcg', 'rank:map'
        # tree_method='gpu_hist', 'ct'
        # booster='gbtree', 'gblinear', 'dart'
        # max_bin
        # n_jobs
        # sampling_method='gradient_based'
        # predictor='gpu_predictor'

        # model.fit(x_train, y_train, group_train, verbose=True,
        #           eval_set=[(x_valid, y_valid)], eval_group=[group_valid])
        # pred = model.predict(x_test)

    def fit(self, batches):

        self.training = True
        users_emb, items_emb = self.representation
        for data in tqdm(batches,
                         desc='train batches',
                         leave=False,
                         dynamic_ncols=True,
                         disable=self.slurm):

            users = data.to(self.device)

            # data = data.to(self.device)
            # users, items = data[:, 0], data[:, 1:]

            y_true, batch, groups = [], [], []

            # sampling the data correctly
            for user in users:
                negatives = self.subtract_tensor_as_set(self.all_items, self.positive_lists[user]['tensor'])
                positives = self.positive_lists[user]['tensor']
                batch.append(torch.cat([positives, negatives]))
                y_true += [1] * len(positives) + [0] * len(negatives)
                groups.append(len(positives) + len(negatives))

            batch = torch.stack(batch)
            vectors = self.get_vectors(users_emb[users],
                                       items_emb[batch],
                                       users,
                                       batch)

            features = self.get_features_batchwise_xgboost(vectors).squeeze()
            features = features.reshape((-1, len(self.feature_names)))
            self.tree.fit(features.detach().cpu().numpy(), y_true, group=groups, verbose=True)

        self.evaluate()
        self.checkpoint(-1)

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        vectors = self.get_vectors(users_emb, items_emb, users, list(range(self.n_items)))
        v = self.get_features_batchwise(vectors).detach().cpu()
        shape = v.shape
        results = self.tree.predict(v.reshape(-1, v.shape[-1])).reshape(shape[:2])
        return torch.from_numpy(results)

    def get_features_batchwise_xgboost(self, vectors):
        '''
            batchwise (all-to-all) calculation of features for top layer:
            vectors['users_emb'].shape = (batch_size, emb_size)
            vectors['items_emb'].shape = (n_items, emb_size)
        '''
        return torch.cat([
            (vectors['users_emb'].unsqueeze(1) @ vectors['items_emb'].transpose(1, 2)).unsqueeze(-1),
            (vectors['users_reviews'].unsqueeze(1) @ vectors['items_reviews'].transpose(1, 2)).unsqueeze(-1),
            (vectors['users_desc'].unsqueeze(1) @ vectors['items_desc'].transpose(1, 2)).unsqueeze(-1),
            (vectors['users_reviews'].unsqueeze(1) @ vectors['items_desc'].transpose(1, 2)).unsqueeze(-1),
            (vectors['users_desc'].unsqueeze(1) @ vectors['items_reviews'].transpose(1, 2)).unsqueeze(-1),
            (vectors['user_gnn_reviews'].unsqueeze(1) @ vectors['item_gnn_reviews'].transpose(1, 2)).unsqueeze(-1),
            (vectors['user_gnn_desc'].unsqueeze(1) @ vectors['item_gnn_desc'].transpose(1, 2)).unsqueeze(-1),
        ], axis=-1)

    def subtract_tensor_as_set(self, t1, t2):
        '''
            quickly subtracts elements of the second tensor from
            the first tensor as if they were sets.

            copied from stackoverflow. no clue how this works
        '''
        return t1[(t2.repeat(t1.shape[0], 1).T != t1).T.prod(1) == 1].type(torch.int64)

    def score_pairwise_adv(self, users_emb, items_emb, *args):
        '''
            each user has a batch with corresponding items,
            calculate scores for all items for those items:
            users_emb.shape = (batch_size, emb_size)
            items_emb.shape = (batch_size, 1000, emb_size)
        '''
        return torch.matmul(users_emb.unsqueeze(1), items_emb.transpose(1, 2)).squeeze()
