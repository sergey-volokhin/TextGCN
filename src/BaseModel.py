import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.optim as opt
from torch import nn
from tqdm.auto import tqdm, trange


class BaseModel(nn.Module, ABC):
    '''
    meta class with objective-agnostic utility functions:
        - training
        - saving
        - loading
        - logging metrics
    '''

    def __init__(self, config, dataset):
        super().__init__()
        self._copy_params(config)
        self._copy_dataset_params(dataset)
        self._add_vars(config)
        self.to(self.device)

    def _copy_params(self, config):
        self.lr = config.lr
        self.to_save = config.save
        self.quiet = config.quiet
        self.epochs = config.epochs
        self.logger = config.logger
        self.device = config.device
        self.patience = config.patience
        self.save_path = config.save_path
        self.evaluate_every = config.evaluate_every
        self.slurm = config.slurm or config.quiet

    def _copy_dataset_params(self, dataset):
        ...

    def _add_vars(self, config):
        ''' add remaining variables '''
        self.last_eval_epoch = -1  # last epoch at which evaluation was performed

    def fit(self, batches):
        ''' training function '''

        self.optimizer = opt.Adam(self.parameters(), lr=self.lr)
        for epoch in trange(1, self.epochs + 1, desc='epochs', disable=self.quiet, dynamic_ncols=True):
            self.train()
            self._loss_values = defaultdict(float)
            epoch_loss = 0
            for data in tqdm(batches,
                             desc='train batches',
                             dynamic_ncols=True,
                             leave=False,
                             disable=self.slurm):
                self.optimizer.zero_grad()
                batch_loss = self.get_loss(data)
                epoch_loss += batch_loss
                batch_loss.backward()
                self.optimizer.step()

            if epoch % self.evaluate_every:
                continue

            self.last_eval_epoch = epoch
            results = self.evaluate()
            self.metrics_log.update(results)

            if self.metrics_log.this_epoch_best():
                self.logger.info(f"Epoch {epoch}: {' '.join([f'{k} = {v:.4f}' for k,v in self._loss_values.items()])}")
                self.metrics_log.log()

            if self.to_save:
                self.save()

            if self.metrics_log.should_stop():
                self.logger.warning(f'Early stopping triggerred at epoch {epoch}')
                break

        if self.last_eval_epoch != self.epochs and self.to_save:
            self.save()

        self.metrics_log.print_best_results()

    def save(self):
        ''' save current model and update the best one '''
        latest_checkpoint_path = os.path.join(self.save_path, 'latest_checkpoint.pkl')
        torch.save(self.state_dict(), latest_checkpoint_path)
        if self.metrics_log.this_epoch_best():
            self.logger.info(f'Updating best model at epoch {self.last_eval_epoch}')
            shutil.copyfile(latest_checkpoint_path, os.path.join(self.save_path, 'best.pkl'))

    def load(self, path):
        assert path is not None, 'No model path to load provided'
        if os.path.isdir(path):
            path = os.path.join(path, 'best.pkl')
        self.logger.info(f'Loading model {path}')
        self.load_state_dict(torch.load(path, map_location=self.device))

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        ''' calculate and report metrics for val/test users against predictions '''

    @torch.no_grad()
    @abstractmethod
    def predict(self, *args, **kwargs):
        ''' return predictions for provided users or dataframe '''

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
