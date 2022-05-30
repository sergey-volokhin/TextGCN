import random

import torch
from tqdm import tqdm, trange
from xgboost import XGBRanker

from advanced_sampling import AdvSamplModel, AdvSamplDataset
from ltr_models import LTRBase, LTRDataset


class XGBoostDataset(LTRDataset, AdvSamplDataset):
    pass


class LTRXGBoost(LTRBase, AdvSamplModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.score_batchwise = self.score_batchwise_ltr

    def _setup_layers(self, args):
        self.tree = XGBRanker(verbosity=1,
                              objective='rank:pairwise',
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

        for epoch in trange(1, self.epochs + 1, desc='epochs', disable=self.quiet):
            self.train()
            self.training = True
            users_emb, items_emb = self.representation
            for data in tqdm(batches,
                             desc='train batches',
                             leave=False,
                             dynamic_ncols=True,
                             disable=self.slurm):

                data = data.to(self.device)
                users, items = data[:, 0], data[:, 1:]
                rankings = self.score_pairwise_adv(users_emb[users], items_emb[items], users)

                y_true, batch, groups = [], [], []

                # sampling the data correctly
                for user, u_items, rank in zip(users, items, rankings):
                    u_items = u_items[rank.argsort(descending=True)]  # sort items by score to select highest rated
                    negatives = self.subtract_tensor_as_set(u_items, self.positive_lists[user]['tensor'])[:max(self.k)]
                    positives = self.positive_lists[user]['list']
                    positives = torch.tensor(random.sample(positives, min(self.pos_samples, len(positives)))).to(self.device)
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
                self.tree.fit(features.detach().cpu().numpy(), y_true, group=groups)

            if epoch % self.evaluate_every:
                continue

            self.evaluate()
            self.checkpoint(epoch)

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
