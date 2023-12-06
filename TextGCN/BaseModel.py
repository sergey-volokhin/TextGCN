import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.optim as opt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from .utils import calculate_metrics, early_stop


class BaseModel(nn.Module, ABC):
    ''' meta class with model-agnostic utility functions '''

    def __init__(self, params, dataset):
        super().__init__()
        self._copy_params(params)
        self._copy_dataset_params(dataset)
        self._add_vars(params)
        self.to(self.device)

    def _copy_params(self, params):
        self.k = params.k
        self.lr = params.lr
        self.uid = params.uid
        self.save = params.save
        self.quiet = params.quiet
        self.epochs = params.epochs
        self.logger = params.logger
        self.device = params.device
        self.save_path = params.save_path
        self.batch_size = params.batch_size
        self.evaluate_every = params.evaluate_every
        self.slurm = params.slurm or params.quiet

    def _copy_dataset_params(self, dataset):
        self.true_test_lil = dataset.true_test_lil
        self.train_user_dict = dataset.train_user_dict
        self.test_users = np.sort(dataset.test_df.user_id.unique())  # ids of people from test set
        self.user_mapping_dict = dict(dataset.user_mapping[['remap_id', 'org_id']].values)  # internal id -> real id
        self.item_mapping_dict = dict(dataset.item_mapping[['remap_id', 'org_id']].values)  # internal id -> real id

    def _add_vars(self, params):
        ''' add remaining variables '''
        self.metrics_log = defaultdict(lambda: defaultdict(list))
        self.metrics = ['recall', 'precision', 'hit', 'ndcg', 'f1']
        self.early_stop_mode = 'max'  # bigger metrics is better
        self.eval_epochs = []  # epochs at which evaluation was performed
        self.training = False
        self.writer = SummaryWriter(self.save_path) if params.tensorboard else False

    def fit(self, batches):
        ''' training function '''

        self.optimizer = opt.Adam(self.parameters(), lr=self.lr)
        for epoch in trange(1, self.epochs + 1, desc='epochs', disable=self.quiet, dynamic_ncols=True):
            self.train()
            self.training = True
            self._loss_values = defaultdict(float)
            epoch_loss = 0
            for data in tqdm(batches,
                             desc='train batches',
                             dynamic_ncols=True,
                             leave=False,
                             disable=self.slurm):
                self.optimizer.zero_grad()
                batch_loss = self.get_loss(data)
                assert not batch_loss.isnan(), f'loss is NA at epoch {epoch}'
                epoch_loss += batch_loss
                batch_loss.backward()
                self.optimizer.step()

            if self.writer:
                self.writer.add_scalar('training loss', epoch_loss, epoch)

            if epoch % self.evaluate_every:
                continue

            self.logger.info(f"Epoch {epoch}: {' '.join([f'{k} = {v:.4f}' for k,v in self._loss_values.items()])}")
            results = self.evaluate()
            self._log_metrics(epoch, results)
            self.print_metrics(results)

            if self.save:
                self.checkpoint()

            if len(self.eval_epochs) > 2 and early_stop(self.metrics_log, mode=self.early_stop_mode):
                self.logger.warning(f'Early stopping triggerred at epoch {epoch}')
                break

        if self.eval_epochs[-1] != self.epochs and self.save:
            self.checkpoint()
        if self.writer:
            self.writer.close()

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
        return calculate_metrics(predictions, self.k)

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

    def checkpoint(self):
        ''' save current model and update the best one '''
        latest_checkpoint_path = os.path.join(self.save_path, 'latest_checkpoint.pkl')
        torch.save(self.state_dict(), latest_checkpoint_path)
        if max(self.metrics_log[self.k[0]]['recall']) <= self.metrics_log[self.k[0]]['recall'][-1]:
            self.logger.info(f'Updating best model at epoch {self.eval_epochs[-1]}')
            shutil.copyfile(latest_checkpoint_path, os.path.join(self.save_path, 'best.pkl'))

    def load(self, path):
        assert path is not None, 'No model path to load provided'
        if os.path.isdir(path):
            path = os.path.join(path, 'best.pkl')
        self.logger.info(f'Loading model {path}')
        self.load_state_dict(torch.load(path, map_location=self.device))

    def _log_metrics(self, epoch, results):
        ''' update metrics logger with new results '''
        self.eval_epochs.append(epoch)
        for k in results:
            for metric in results[k]:
                self.metrics_log[k][metric].append(results[k][metric])

    def print_metrics(self, results):
        ''' show metrics in the log '''
        self.logger.info(' ' * 11 + ''.join([f'@{i:<6}' for i in results]))
        for metric in self.metrics:
            self.logger.info(f'{metric:11}' + ' '.join([f'{results[i][metric]:.4f}' for i in results]))

    @property
    @abstractmethod
    def representation(self, *args, **kwargs):
        ''' return a tuple of user and item embeddings '''

    @abstractmethod
    def get_loss(self, *args, **kwargs):
        ''' get total loss per batch of users '''

    @abstractmethod
    def score_pairwise(self, *args, **kwargs):
        '''
        calculate predicted user-item scores for a list of pairs (u, i):
            users_emb.shape == items_emb.shape
        '''

    @abstractmethod
    def score_batchwise(self, *args, **kwargs):
        '''
        calculate predicted user-item scores batchwise (all-to-all):
            users_emb.shape == (batch_size, emb_size)
            items_emb.shape == (n_items, emb_size)
        '''
