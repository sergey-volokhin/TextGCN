import os
from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .BaseModel import BaseModel


class ScoringModel(BaseModel):
    '''
    Base class for models that predict ratings for user-item pairs
    uses MSE loss
    when predicting, score only user-item pairs from test_df
    evaluation using MSE, MAE
    '''

    def _copy_params(self, params):
        super()._copy_params(params)
        self.classification = params.classification

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.scalers = dataset.scalers
        self.test_df = dataset.test_df
        self.test_scores = torch.from_numpy(self.test_df['rating'].values).to(self.device)
        if hasattr(dataset, '_actual_test_df'):
            self._actual_test_df = dataset._actual_test_df
            self._actual_test_scores = torch.from_numpy(self._actual_test_df['rating'].values).to(self.device)

    def _add_vars(self, *args, **kwargs):
        super()._add_vars(*args, **kwargs)
        self.metrics = ['MSE', 'MAE']
        self.early_stop_mode = 'min'  # lower metrics is better
        self.loss = nn.MSELoss()
        self.best_valid_mse = np.inf
        self.best_valid_mae = np.inf
        self.best_test_mse = np.inf
        self.best_test_mae = np.inf

    # def _normalize_ratings
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.logger.info(f'Best valid MSE: {self.best_valid_mse:.4f}')
        self.logger.info(f'Best valid MAE: {self.best_valid_mae:.4f}')
        self.logger.info(f'Best test MSE:  {self.best_test_mse:.4f}')
        self.logger.info(f'Best test MAE:  {self.best_test_mae:.4f}')

    def _unnormalize_ratings(self, ratings_df):
        ''' takes a dataframe with columns ['user_id', 'rating'] and returns a tensor of unnormalized ratings '''

        def unscale(group):
            scaler = self.scalers[group.name]
            return group.values * scaler['scale'] + scaler['mean']

        result = torch.from_numpy(np.concatenate(ratings_df.groupby('user_id')['rating'].apply(unscale).values)).to(self.device)

        if self.classification:
            return torch.round(result)
        return result

    def get_loss(self, data):
        users, items, ratings = data.t()
        users = users.long()
        items = items.long()
        mse_loss = self.mse_loss(users, items, ratings.float())
        reg_loss = self.reg_loss(users, items)
        self._loss_values['mse'] += mse_loss
        self._loss_values['reg'] += reg_loss
        return mse_loss + reg_loss

    def mse_loss(self, users, items, ratings):
        '''mean squared error loss'''
        users_emb, items_emb = self.representation
        predictions = self.score_pairwise(users_emb[users], items_emb[items], users, items)
        return self.loss(predictions.flatten(), ratings)

    @abstractmethod
    def reg_loss(self, *args, **kwargs):
        ''' regularization L2 loss '''

    @torch.no_grad()
    def evaluate(self):
        self.eval()
        self.training = False
        predictions = self.predict()
        scores = {'': {'MSE': F.mse_loss(predictions, self.test_scores).round(decimals=4),
                       'MAE': F.l1_loss(predictions, self.test_scores).round(decimals=4)}}
        if hasattr(self, '_actual_test_df') and scores['']['MSE'] < self.best_valid_mse:
            self.best_valid_mse = scores['']['MSE']
            self.best_valid_mae = scores['']['MAE']
            test_scores = self.predict(self._actual_test_df)
            self.best_test_mse = F.mse_loss(test_scores, self._actual_test_scores).round(decimals=4)
            self.best_test_mae = F.l1_loss(test_scores, self._actual_test_scores).round(decimals=4)
        return scores

    @torch.no_grad()
    def predict(self, df=None, save: bool = False, *args, **kwargs):
        '''
        scores all user-item pairs from test_df
        '''
        self.training = False

        if df is None:
            df = self.test_df

        users_emb, items_emb = self.representation
        users = torch.tensor(df['user_id'].values)
        items = torch.tensor(df['asin'].values)
        df['predicted_score'] = self.score_pairwise(users_emb[users], items_emb[items], users, items).cpu().numpy()
        predictions = self._unnormalize_ratings(df[['user_id', 'predicted_score']].rename(columns={'predicted_score': 'rating'}))

        if save:
            df['predicted_score'] = predictions.cpu().numpy()
            df.to_csv(os.path.join(self.save_path, 'predicted_scores.tsv'), sep='\t', index=False)
            df.drop(columns=['predicted_score'], inplace=True)

        return predictions

    def _last_epoch_best(self):
        return self.best_valid_mse == self.metrics_log['']['MSE'][-1]