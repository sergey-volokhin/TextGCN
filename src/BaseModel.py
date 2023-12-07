import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.optim as opt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from .utils import early_stop


class BaseModel(nn.Module, ABC):
    '''
    meta class with objective-agnostic utility functions:
        - training
        - saving
        - loading
        - logging metrics
    '''

    def __init__(self, params, dataset):
        super().__init__()
        self._copy_params(params)
        self._copy_dataset_params(dataset)
        self._add_vars(params)
        self.to(self.device)

    def _copy_params(self, params):
        self.lr = params.lr
        self.to_save = params.save
        self.quiet = params.quiet
        self.epochs = params.epochs
        self.logger = params.logger
        self.device = params.device
        self.save_path = params.save_path
        self.evaluate_every = params.evaluate_every
        self.slurm = params.slurm or params.quiet

    def _copy_dataset_params(self, dataset):
        ...

    def _add_vars(self, params):
        ''' add remaining variables '''
        self.training = False
        self.metrics_log = defaultdict(lambda: defaultdict(list))
        self.eval_epochs = []  # epochs at which evaluation was performed
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

            if self.to_save:
                self.save()

            if len(self.eval_epochs) > 2 and early_stop(self.metrics_log, mode=self.early_stop_mode):
                self.logger.warning(f'Early stopping triggerred at epoch {epoch}')
                break

        if self.eval_epochs[-1] != self.epochs and self.to_save:
            self.save()
        if self.writer:
            self.writer.close()

    def save(self):
        ''' save current model and update the best one '''
        latest_checkpoint_path = os.path.join(self.save_path, 'latest_checkpoint.pkl')
        torch.save(self.state_dict(), latest_checkpoint_path)
        if self._last_epoch_best():
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

    @torch.no_grad()
    @abstractmethod
    def evaluate(self):
        ''' calculate and report metrics for test users against predictions '''

    @torch.no_grad()
    @abstractmethod
    def predict(self, *args, **kwargs):
        ''' return predictions for given users '''

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

    @abstractmethod
    def _last_epoch_best(self):
        ''' check if the last epoch was the best one '''
