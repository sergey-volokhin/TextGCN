import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm, trange

from base_model import BaseModel
from kg_models import DatasetKG
from reviews_models import DatasetReviews
from utils import early_stop


class LTRDataset(DatasetKG, DatasetReviews):

    def __init__(self, args):
        super().__init__(args)
        self._get_users_as_avg_reviews()

    def _get_users_as_avg_reviews(self):
        ''' use average of reviews to represent users '''
        user_text_embs = {}
        for user, group in self.top_med_reviews.groupby('user_id')['vector']:
            user_text_embs[user] = torch.tensor(group.values.tolist()).mean(axis=0)
        self.users_as_avg_reviews = torch.stack(self.user_mapping['remap_id'].map(
            user_text_embs).values.tolist()).to(self.device)


class LTR(BaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        if args.model == 'ltr_reviews':
            self.score = self.score_reviews
        elif args.model == 'ltr_kg':
            self.score = self.score_kg
        else:
            raise AttributeError(f'wrong LTR model name: "{args.model}"')

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def score_reviews(self, users, items, users_emb, item_emb):
        user_part = torch.cat([users_emb, self.users_as_avg_reviews[users]], axis=1)
        item_part = torch.cat([item_emb, self.items_as_avg_reviews[items]], axis=1)
        return torch.sum(torch.mul(user_part, item_part), dim=1)

    def score_kg(self, users, items, users_emb, item_emb):
        user_part = torch.cat([users_emb, self.users_as_avg_reviews[users]], axis=1)
        item_part = torch.cat([item_emb, self.items_as_desc[items]], axis=1)
        return torch.sum(torch.mul(user_part, item_part), dim=1)


class LTRSimple(BaseModel):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def predict(self, save=False):
        '''
            using new formula:
            :math:`\sigma((\mathbf{e}_{u,gnn}||\mathbf{e}_{u,text})\cdot(\mathbf{e}_{neg,gnn}||\mathbf{e}_{neg,text}) - \\
            (\mathbf{e}_{u,gnn}||\mathbf{e}_{u,text})\cdot(\mathbf{e}_{pos, gnn}||\mathbf{e}_{pos,text}))`
        '''

        y_probs, y_pred = [], []

        with torch.no_grad():
            users_emb, items_emb = self.representation

            # append textual representation of items to their GNN vectors
            full_items_emb = torch.cat([items_emb, self.items_as_avg_reviews], axis=1)

            for batch_users in tqdm(self.test_batches,
                                    desc='batches',
                                    leave=False,
                                    dynamic_ncols=True,
                                    disable=self.slurm):

                # append textual representation of users to their GNN vectors
                full_batch_user_emb = torch.cat([users_emb[batch_users],
                                                 self.users_as_avg_reviews[batch_users]], axis=1)

                rating = torch.matmul(full_batch_user_emb, full_items_emb.t())
                exploded = self.train_user_dict[batch_users].explode()
                rating[(exploded.index - exploded.index.min()).tolist(), exploded.tolist()] = np.NINF
                probs, rank_indices = torch.topk(rating, k=max(self.k))
                y_pred += rank_indices.tolist()
                y_probs += probs.round(decimals=4).tolist()

        predictions = pd.DataFrame.from_dict({'y_true': self.true_test_lil,
                                              'y_pred': y_pred,
                                              'scores': y_probs})
        if save:
            predictions.to_csv(f'{self.save_path}/predictions.tsv', sep='\t', index=False)
            self.logger.info(f'Predictions are saved in `{self.save_path}/predictions.tsv`')
        return predictions


class LTRLinear(BaseModel):

    def __init__(self, args, dataset) -> None:
        super().__init__(args, dataset)

        self.save_path = f'runs/{os.path.basename(os.path.dirname(args.data))}/{args.uid}'

        self.distill_text = nn.Linear(384, args.emb_size).to(self.device)  # TODO remove magic dim of embeddings
        self.linear = nn.Linear(args.emb_size * 4, 1).to(self.device)  # gnn_u + text_distil_u + gnn_i + text_distil_i
        self.loss = nn.BCEWithLogitsLoss()

        # self.get_text_item_repr = self.get_item_reviews_user  # not gonna work
        # self.get_text_item_repr = self.get_items_as_desc
        self.get_text_item_repr = self.get_item_reviews_mean
        self.get_text_user_repr = self.get_user_reviews_mean

        # ''' represent items using their descriptions or mean of reviews '''
        # if :
        #     self.get_text_item_repr = self.get_items_as_desc
        # elif:
        #     self.get_text_item_repr = self.get_item_reviews_mean

        # ''' represent users by the mean of their reviews '''
        # if :
        #     self.get_text_user_repr = self.get_item_reviews_user

    def _copy_args(self, args):
        super()._copy_args(args)
        self.load = args.load

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.items_as_desc = dataset.items_as_desc
        self.reviews_vector = dataset.reviews_vector
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews

    def fit(self, batches):

        self.epochs = 1
        if not self.load:
            super().fit(batches)
        self.epochs = 500

        self._get_optimizer()
        for epoch in trange(1, self.epochs + 1, desc='ltr epochs', disable=self.quiet):
            self.train()
            self.training = True
            total_loss = 0
            for data in tqdm(batches,
                             desc='batches',
                             leave=False,
                             dynamic_ncols=True,
                             disable=self.slurm):
                self.optimizer.zero_grad()
                loss = self.get_ltr_loss(*data.to(self.device).t())
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            self.scheduler.step(total_loss)

            if epoch % self.evaluate_every:
                continue

            self.logger.info(f'Epoch {epoch}: loss = {total_loss:.4f}')
            self.evaluate_ltr()
            self.checkpoint(epoch)

            if early_stop(self.metrics_logger):
                self.logger.warning(f'Early stopping triggerred at epoch {epoch}')
                break

    def get_user_reviews_mean(self, u):
        ''' represent users with mean of their reviews '''
        return self.users_as_avg_reviews[u]

    def get_item_reviews_mean(self, u, i):
        ''' represent items with mean of their reviews '''
        return self.items_as_avg_reviews[i]

    def get_items_as_desc(self, u, i):
        ''' represent items with their description '''
        return self.items_as_desc[i]

    def get_item_reviews_user(self, u, i):
        ''' represent items with the review of corresponding user '''
        df = self.reviews_vector.loc[torch.stack([i, u], axis=1).tolist()]
        return torch.tensor(df.values.tolist()).to(self.device)

    def predict_ltr(self, save=False):
        '''
            returns a dataframe with predicted and true items for each test user:
            pd.DataFrame(columns=['y_pred', 'y_true'])

            using formula:
            ..math::`\sigma(\mathbf{e}_{u,gnn}\mathbf{e}_{neg,gnn} - \mathbf{e}_{u,gnn}\mathbf{e}_{pos,gnn}).
        '''

        y_probs, y_pred = [], []

        with torch.no_grad():
            users_emb, items_emb = self.representation
            for batch_users in tqdm(self.test_batches,
                                    desc='batches',
                                    leave=False,
                                    dynamic_ncols=True,
                                    disable=self.slurm):

                gnn_u = users_emb[batch_users]
                gnn_i = items_emb

                text_u = self.get_text_user_repr(batch_users)
                text_i = self.get_text_item_repr(batch_users, range(self.n_items))

                distil_u = self.distill_text(text_u)
                distil_i = self.distill_text(text_i)

                user_vectors = torch.cat([gnn_u, distil_u], axis=1)
                item_vectors = torch.cat([gnn_i, distil_i], axis=1)

                rating = []
                for user in user_vectors:
                    user_repeated = user.unsqueeze(0).repeat(len(item_vectors), 1)
                    vector = torch.cat([user_repeated, item_vectors], axis=1)
                    rating.append(self.linear(vector).squeeze())
                rating = torch.stack(rating)

                '''
                    set scores for train items to be -inf so we don't recommend them.
                    we subtract exploded.index.min because rating matrix only has
                    batch_size users, so it starts with 0, while index has users' real indices
                '''
                exploded = self.train_user_dict[batch_users].explode()
                rating[(exploded.index - exploded.index.min()).tolist(), exploded.tolist()] = np.NINF

                # select top-k items with highest ratings
                probs, rank_indices = torch.topk(rating, k=max(self.k))
                y_pred += rank_indices.tolist()
                y_probs += probs.round(decimals=4).tolist()  # TODO: rounding doesn't work for some reason

        predictions = pd.DataFrame.from_dict({'y_true': self.true_test_lil,
                                              'y_pred': y_pred,
                                              'scores': y_probs})
        if save:
            predictions.to_csv(f'{self.save_path}/predictions.tsv', sep='\t', index=False)
            self.logger.info(f'Predictions are saved in `{self.save_path}/predictions.tsv`')
        return predictions

    def evaluate_ltr(self):
        ''' calculate and report metrics for test users against predictions '''
        self.eval()
        self.training = False
        predictions = self.predict_ltr()
        results = {i: np.zeros(len(self.k)) for i in self.metrics}
        for col in predictions.columns:
            predictions[col] = predictions[col].apply(np.array)
        for k in self.k:
            predictions[f'intersection_{k}'] = predictions.apply(
                lambda row: np.intersect1d(row['y_pred'][:k], row['y_true']), axis=1)

        ''' get metrics per user & aggregates '''
        n_users = len(self.true_test_lil)
        for row in predictions.itertuples(index=False):
            r = self.test_one_user(row)
            for metric in r:
                results[metric] += r[metric]
        for metric in results:
            results[metric] /= n_users

        ''' show metrics in log '''
        self.logger.info(' ' * 11 + ''.join([f'@{i:<6}' for i in self.k]))
        for i in results:
            self.metrics_logger[i] = np.append(self.metrics_logger[i], [results[i]], axis=0)
            self.logger.info(f'{i:11}' + ' '.join([f'{j:.4f}' for j in results[i]]))
        self.save_progression()
        return results

    def get_ltr_loss(self, users, pos, neg):

        # user
        gnn_u = self.embedding_user(users)
        text_u = self.get_text_user_repr(users)
        distil_u = self.distill_text(text_u)

        # positive items
        gnn_p = self.embedding_item(pos)
        text_p = self.get_text_item_repr(users, pos)
        distil_p = self.distill_text(text_p)

        # negative items
        gnn_n = self.embedding_item(neg)
        text_n = self.get_text_item_repr(users, neg)
        distil_n = self.distill_text(text_n)

        vector = torch.cat([gnn_u, distil_u, gnn_p, distil_p], axis=1)
        logits = self.linear(vector)

        loss = self.loss(logits, torch.ones_like(logits))
        vector = torch.cat([gnn_u, distil_u, gnn_n, distil_n], axis=1)

        logits = self.linear(vector)
        loss += self.loss(logits, torch.zeros_like(logits))
        return loss
