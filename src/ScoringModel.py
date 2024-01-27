import os

import numpy as np
import torch
from torch import nn

from .BaseModel import BaseModel
from .utils import calculate_scoring_metrics as calculate_metrics
from .MetricsTracker import ScoringMetricsTracker


class ScoringModel(BaseModel):
    '''
    Base class for models that predict ratings for user-item pairs
    uses MSE loss
    when predicting, score only user-item pairs from test_df
    evaluation using MSE, MAE
    '''

    def _copy_params(self, config):
        super()._copy_params(config)
        self.reg_lambda = config.reg_lambda
        self.classification = config.classification

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.scalers = dataset.scalers
        self.train_df = dataset.train_df
        self.train_true_score = torch.from_numpy(self.train_df['rating'].values).to(self.device)
        self.val_df = dataset.test_df
        self.val_true_score = torch.from_numpy(self.val_df['rating'].values).to(self.device)
        if hasattr(dataset, '_actual_test_df'):
            self.test_df = dataset._actual_test_df
            self.test_true_score = torch.from_numpy(self.test_df['rating'].values).to(self.device)

    def _add_vars(self, config):
        super()._add_vars(config)
        self.metrics_log = ScoringMetricsTracker(self.logger, self.patience)
        self.loss = nn.MSELoss()

    def _unnormalize_ratings(self, df) -> torch.Tensor:
        ''' takes a dataframe with columns ['user_id', 'rating'] and returns a tensor of unnormalized ratings '''

        def unscale(group):
            scaler = self.scalers[group.name]
            return group.values * scaler['std'] + scaler['mean']

        res = torch.from_numpy(np.concatenate(df.groupby('user_id')['rating'].apply(unscale).values)).to(self.device)

        if self.classification:
            return torch.round(res)
        return res

    def get_loss(self, data):
        users, items, ratings = data.t()
        users = users.long()
        items = items.long()
        mse_loss = self.mse_loss(users, items, ratings.float())
        reg_loss = self.reg_lambda * self.reg_loss(users, items)
        self._loss_values['mse'] += mse_loss
        self._loss_values['reg'] += reg_loss
        return mse_loss + reg_loss

    def mse_loss(self, users, items, ratings):
        '''mean squared error loss'''
        users_emb, items_emb = self.forward()
        predictions = self.score_pairwise(users_emb[users], items_emb[items], users, items)
        return self.loss(predictions.flatten(), ratings)

    @torch.no_grad()
    def evaluate(self):
        ''' returns a dict of metrics for val and test sets: {"split metric": value} '''
        self.eval()
        val_preds = self.predict()

        results = {f'train_{k}': v for k, v in calculate_metrics(self.predict(self.train_df), self.train_true_score).items()}
        results.update({f'valid_{k}': v for k, v in calculate_metrics(val_preds, self.val_true_score).items()})

        if hasattr(self, 'test_df') and self.metrics_log._is_better(results):
            test_preds = self.predict(df=self.test_df)
            results.update({f'test_{k}': v for k, v in calculate_metrics(test_preds, self.test_true_score).items()})
        return results

    @torch.no_grad()
    def predict(self, df=None, save: bool = False, *args, **kwargs) -> torch.Tensor:
        ''' scores all user-item pairs from val_df or from provided df '''
        self.eval()

        if df is None:
            df = self.val_df

        users_emb, items_emb = self.forward()
        users = torch.tensor(df['user_id'].values)
        items = torch.tensor(df['asin'].values)
        df['pred_score'] = self.score_pairwise(users_emb[users], items_emb[items], users, items).cpu().numpy()
        predictions = self._unnormalize_ratings(df[['user_id', 'pred_score']].rename(columns={'pred_score': 'rating'}))
        predictions = torch.clamp(predictions, min=1, max=5)

        if save:
            df['pred_score'] = predictions.cpu().numpy()
            df.to_csv(os.path.join(self.save_path, 'predicted_scores.tsv'), sep='\t', index=False)

        return predictions
