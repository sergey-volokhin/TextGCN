import os
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .BaseModel import BaseModel
from .utils import calculate_ranking_metrics


class RankingModel(BaseModel):
    '''
    Base class for models that rank items for each user
    more specifically, predict existance of links between users and items
    use BPR loss
    when predicting, rank all the items per user and return top-k
    evaluation using Recall, Precision, Hit Rate, NDCG, F1
    '''

    def _copy_params(self, params):
        super()._copy_params(params)
        self.k = params.k
        self.batch_size = params.batch_size

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.true_test_lil = dataset.true_test_lil
        self.train_user_dict = dataset.train_user_dict
        self.test_users = np.sort(dataset.test_df.user_id.unique())  # ids of people from test set
        self.user_mapping_dict = dict(dataset.user_mapping[['remap_id', 'org_id']].values)  # internal id -> real id
        self.item_mapping_dict = dict(dataset.item_mapping[['remap_id', 'org_id']].values)  # internal id -> real id

    def _add_vars(self, *args, **kwargs):
        super()._add_vars(*args, **kwargs)
        self.activation = F.selu  # F.softmax
        self.metrics = ['recall', 'precision', 'hit', 'ndcg', 'f1']
        self.early_stop_mode = 'max'  # bigger metrics is better

    def get_loss(self, data):
        users, pos, *negs = data.to(self.device).t()
        bpr_loss = self.bpr_loss(users, pos, negs)
        reg_loss = self.reg_loss(users, pos, negs)
        self._loss_values['bpr'] += bpr_loss
        self._loss_values['reg'] += reg_loss
        return bpr_loss + reg_loss

    def bpr_loss(self, users, pos, negs):
        ''' Bayesian Personalized Ranking pairwise loss '''
        users_emb, items_emb = self.representation
        users_emb = users_emb[users]
        pos_scores = self.score_pairwise(users_emb, items_emb[pos], users, pos)
        loss = 0
        for neg in negs:  # todo: vectorize
            neg_scores = self.score_pairwise(users_emb, items_emb[neg], users, neg)
            loss += torch.mean(self.activation(neg_scores - pos_scores))
        return loss / len(negs)

    @abstractmethod
    def reg_loss(self, *args, **kwargs):
        ''' regularization L2 loss '''

    @torch.no_grad()
    def predict(self, users, save: bool = False, with_scores: bool = False):
        '''
        returns a list of lists with predicted items for given list of user_ids
            optionally with probabilities
        '''
        # todo predict using unmapped ids

        self.training = False
        y_probs, y_pred = [], []
        batches = [users[j:j + self.batch_size] for j in range(0, len(users), self.batch_size)]
        users_emb, items_emb = self.representation
        for batch_users in tqdm(batches,
                                desc='predict batches',
                                leave=False,
                                dynamic_ncols=True,
                                disable=self.slurm):

            rating = self.score_batchwise(users_emb[batch_users], items_emb, batch_users)

            ''' set scores for train items to be -inf so we don't recommend them. '''
            exploded = self.train_user_dict[batch_users].reset_index(drop=True).explode()
            rating[exploded.index, exploded.tolist()] = np.NINF

            ''' select top-k items with highest ratings '''
            probs, rank_indices = torch.topk(rating, k=max(self.k))
            y_pred.append(rank_indices)
            y_probs.append(probs.round(decimals=4))  # TODO: rounding doesn't work for some reason

        predictions = torch.cat(y_pred).tolist()
        scores = torch.cat(y_probs).tolist()

        if save:
            predictions_unmapped = [[self.item_mapping_dict[i] for i in row] for row in predictions]
            users_unmapped = [self.user_mapping_dict[u] for u in users]
            pred_df = pd.DataFrame({'user_id': users_unmapped, 'y_pred': predictions_unmapped, 'scores': scores})
            pred_df.to_csv(os.path.join(self.save_path, 'predictions.tsv'), sep='\t', index=False)
            self.logger.info(f"Predictions are saved in `{os.path.join(self.save_path, 'predictions.tsv')}`")

        if with_scores:
            return predictions, scores
        return predictions

    @torch.no_grad()
    def evaluate(self):
        ''' calculate and report metrics for test users against predictions '''
        self.eval()
        self.training = False
        predictions, scores = self.predict(users=self.test_users, with_scores=True)
        predictions = pd.DataFrame.from_dict({
            'user_id': self.test_users,
            'y_true': self.true_test_lil,
            'y_pred': predictions,
            'scores': scores,
        })
        return calculate_ranking_metrics(predictions, self.k)

    def _last_epoch_best(self):
        return max(self.metrics_log[self.k[0]]['recall']) <= self.metrics_log[self.k[0]]['recall'][-1]