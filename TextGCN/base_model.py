import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.optim as opt
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm, trange

from .utils import early_stop, hit, ndcg, precision, recall


class BaseModel(nn.Module):
    '''
    meta class with model-agnostic utility functions
    also works as custom lgcn (vs 'lightgcn' from torch_geometric)
    '''

    def __init__(self, args, dataset) -> None:
        super().__init__()
        self._copy_args(args)
        self._copy_dataset_args(dataset)
        self._init_embeddings(args.emb_size)
        self._add_vars(args)

        self.load_model(args.load)
        self.to(args.device)

    def _copy_args(self, args) -> None:
        self.k = args.k
        self.lr = args.lr
        self.uid = args.uid
        self.save = args.save
        self.quiet = args.quiet
        self.epochs = args.epochs
        self.logger = args.logger
        self.device = args.device
        self.dropout = args.dropout
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.save_path = args.save_path
        self.batch_size = args.batch_size
        self.reg_lambda = args.reg_lambda
        self.evaluate_every = args.evaluate_every
        self.neg_samples = args.neg_samples
        self.slurm = args.slurm or args.quiet
        if args.single:
            self.layer_combination = self.layer_combination_single

    def _copy_dataset_args(self, dataset) -> None:
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.norm_matrix = dataset.norm_matrix
        self.test_batches = dataset.test_batches
        self.true_test_lil = dataset.true_test_lil
        self.train_user_dict = dataset.train_user_dict
        self.test_users = np.sort(dataset.test_df.user_id.unique())  # ids of people from test set
        self.user_mapping_dict = dict(dataset.user_mapping[['remap_id', 'org_id']].values)   # internal id -> real id
        self.item_mapping_dict = dict(dataset.item_mapping[['remap_id', 'org_id']].values)   # internal id -> real id

    def _init_embeddings(self, emb_size: int) -> None:
        ''' randomly initialize entity embeddings '''
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=emb_size).to(self.device)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=emb_size).to(self.device)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def _add_vars(self, args) -> None:
        ''' add remaining variables '''
        self.metrics = ['recall', 'precision', 'hit', 'ndcg', 'f1']
        self.metrics_logger = {i: np.zeros((0, len(self.k))) for i in self.metrics}
        self.training = False

    @property
    def _dropout_norm_matrix(self) -> torch.Tensor:
        ''' drop elements from adj table to help with overfitting '''
        index = self.norm_matrix.indices().t()
        values = self.norm_matrix.values()
        random_index = (torch.rand(len(values)) + (1 - self.dropout)).int().bool()
        index = index[random_index]
        values = values[random_index] / (1 - self.dropout)
        matrix = torch.sparse.FloatTensor(index.t(), values, self.norm_matrix.size())
        return matrix.coalesce().to(self.device)

    @property
    def embedding_matrix(self) -> torch.Tensor:
        ''' 0th layer embedding matrix '''
        return torch.cat([self.embedding_user.weight, self.embedding_item.weight])

    @property
    def representation(self) -> list[torch.Tensor]:
        '''
        aggregate embeddings from neighbors for each layer,
        combine layers into final representations
        '''
        norm_matrix = self._dropout_norm_matrix if self.training else self.norm_matrix
        curent_lvl_emb_matrix = self.embedding_matrix
        node_embed_cache = [curent_lvl_emb_matrix]
        for _ in range(self.n_layers):
            curent_lvl_emb_matrix = self.layer_aggregation(norm_matrix, curent_lvl_emb_matrix)
            node_embed_cache.append(curent_lvl_emb_matrix)
        aggregated_embeddings = self.layer_combination(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def fit(self, batches):
        ''' training function '''

        self.optimizer = opt.Adam(self.parameters(), lr=self.lr)
        for epoch in trange(1, self.epochs + 1, desc='epochs', disable=self.quiet):
            self.train()
            self.training = True
            self._loss_values = defaultdict(float)
            total_loss = 0
            for data in tqdm(batches,
                             desc='train batches',
                             leave=False,
                             dynamic_ncols=True,
                             disable=self.slurm):
                self.optimizer.zero_grad()
                loss = self.get_loss(data)
                assert not loss.isnan(), f'loss is NA at epoch {epoch}'
                total_loss += loss
                loss.backward()
                self.optimizer.step()

            if epoch % self.evaluate_every:
                continue

            self.logger.info(f"Epoch {epoch}: {' '.join([f'{k} = {v:.4f}' for k,v in self._loss_values.items()])}")
            self.evaluate()
            self.checkpoint(epoch)

            if early_stop(self.metrics_logger):
                self.logger.warning(f'Early stopping triggerred at epoch {epoch}')
                break
        else:
            self.checkpoint(self.epochs)
        self.logger.info(f'Full progression of metrics is saved in `{self.progression_path}`')

    def layer_aggregation(self, norm_matrix: torch.Tensor, emb_matrix: torch.Tensor) -> torch.Tensor:
        '''
        aggregate the neighbor's representations
        to get next layer node representation.

        default: normalized sum
        '''
        return torch.sparse.mm(norm_matrix, emb_matrix)

    def layer_combination(self, vectors: list[torch.Tensor]) -> torch.Tensor:
        '''
        combine embeddings from all layers
        into final representation matrix.

        default: mean of all layers
        '''
        return torch.mean(torch.stack(vectors), axis=0)

    def layer_combination_single(self, vectors):
        '''
        only return the last layer representation
        instead of combining all layers
        '''
        return vectors[-1]

    def score_pairwise(
        self,
        users_emb: torch.Tensor,
        items_emb: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        '''
        calculate predicted user-item scores for a list of pairs (u, i):
            users_emb.shape == items_emb.shape
        '''
        return torch.sum(torch.mul(users_emb, items_emb), dim=1)

    def score_batchwise(
        self,
        users_emb: torch.Tensor,
        items_emb: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        '''
        calculate predicted user-item scores batchwise (all-to-all):
            users_emb.shape = (batch_size, emb_size)
            items_emb.shape = (n_items, emb_size)
        '''
        return torch.matmul(users_emb, items_emb.t())

    def get_loss(self, data):
        ''' get total loss per batch of users '''
        users, pos, *negs = data.to(self.device).t()
        return self.bpr_loss(users, pos, negs) + self.reg_loss(users, pos, negs)

    def bpr_loss(
        self,
        users: list[int],
        pos: list[int],
        negs: list[int],
    ):
        ''' Bayesian Personalized Ranking pairwise loss '''
        users_emb, items_emb = self.representation
        users_emb = users_emb[users]
        pos_scores = self.score_pairwise(users_emb, items_emb[pos], users, pos)
        loss = 0
        for neg in negs:
            neg_scores = self.score_pairwise(users_emb, items_emb[neg], users, neg)
            loss += torch.mean(F.selu(neg_scores - pos_scores))
            # loss += torch.mean(F.softmax(neg_scores - pos_scores))
        loss /= len(negs)
        self._loss_values['bpr'] += loss
        return loss

    def reg_loss(
        self,
        users: list[int],
        pos: list[int],
        negs: list[int],
    ):
        ''' regularization L2 loss '''
        loss = self.embedding_user(users).norm(2).pow(2) + self.embedding_item(pos).norm(2).pow(2)
        for neg in negs:
            loss += self.embedding_item(neg).norm(2).pow(2) / len(negs)
        res = self.reg_lambda * loss / len(users) / 2
        self._loss_values['reg'] += res
        return res

    def evaluate(self) -> dict[str, list[float]]:
        ''' calculate and report metrics for test users against predictions '''
        self.eval()
        self.training = False
        predictions, scores = self.predict(self.test_users, with_scores=True)

        predictions = pd.DataFrame.from_dict({
            'user_id': self.test_users,
            'y_true': self.true_test_lil,
            'y_pred': predictions,
            'scores': scores,
        })

        results = self.calculate_metrics(predictions)

        ''' show metrics in log '''
        self.logger.info(' ' * 11 + ''.join([f'@{i:<6}' for i in self.k]))
        for i in results:
            self.metrics_logger[i] = np.append(self.metrics_logger[i], [results[i]], axis=0)
            self.logger.info(f'{i:11}' + ' '.join([f'{j:.4f}' for j in results[i]]))
        self.save_progression()
        return results

    def predict(
        self,
        users: list[int],
        with_scores: bool = False,
        save: bool = False,
    ) -> list | tuple[list, list]:

        '''
        returns a list of lists with predicted items for given list of user_ids
            optionally with probabilities
        '''
        # todo predict using unmapped ids

        self.training = False
        y_probs, y_pred = [], []
        batches = [users[j:j + self.batch_size] for j in range(0, len(users), self.batch_size)]
        with torch.no_grad():
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
            pred_df = pd.DataFrame({'user_id': users_unmapped,
                                    'y_pred': predictions_unmapped,
                                    'scores': scores})
            pred_df.to_csv(f'{self.save_path}/predictions.tsv', sep='\t', index=False)
            self.logger.info(f'Predictions are saved in `{self.save_path}/predictions.tsv`')
        if with_scores:
            return predictions, scores
        return predictions

    def calculate_metrics(self, df: pd.DataFrame) -> dict[str, list[float]]:
        ''' computes all metrics for predictions for all users '''
        result = {i: [] for i in self.metrics}
        df['y_true_len'] = df['y_true'].apply(len)

        ''' calculate intersections of y_pred and y_test '''
        for col in df.columns:
            df[col] = df[col].apply(np.array)

        for k in sorted(self.k):
            df[f'intersection_{k}'] = df.apply(lambda row: np.intersect1d(row['y_pred'][:k], row['y_true']), axis=1)

            df[f'y_pred_{k}'] = df['y_pred'].apply(lambda x: x[:k])
            df['intersecting_len'] = df[f'intersection_{k}'].apply(len)
            rec = recall(df)
            prec = precision(df, k)
            result['recall'].append(rec.mean())
            result['precision'].append(prec.mean())
            result['hit'].append(hit(df).mean())
            result['ndcg'].append(ndcg(df, k).mean())
            numerator = rec * prec * 2
            denominator = rec + prec
            result['f1'].append(np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0).mean(),
            )
        return result

    def load_model(self, load_path: str) -> None:
        ''' load and eval model from file '''
        self.progression_path = f'{self.save_path}/progression.txt'
        if load_path is not None:
            self.logger.info(f'Loading model {load_path}')
            self.load_state_dict(torch.load(load_path, map_location=self.device))
            self.logger.info('Performance of the loaded model:')
            self.evaluate()
            self.metrics_logger = {i: np.zeros((0, len(self.k))) for i in self.metrics}
        else:
            self.logger.info(f'Created model {self.uid}')

    def checkpoint(self, epoch: int) -> None:
        ''' save current model and update the best one '''
        if not self.save:
            return
        torch.save(self.state_dict(), f'{self.save_path}/checkpoint.pkl')
        if self.metrics_logger[self.metrics[0]][:, 0].max() == self.metrics_logger[self.metrics[0]][-1][0]:
            self.logger.info(f'Updating best model at epoch {epoch}')
            os.system(f'cp {self.save_path}/checkpoint.pkl {self.save_path}/best.pkl')
            shutil.copyfile(os.path.join(self.save_path, 'checkpoint.pkl'), os.path.join(self.save_path, 'best.pkl'))

    def save_progression(self) -> None:
        ''' save all scores in one file for clarity '''
        epochs_string, at_string = [' ' * 9], [' ' * 9]
        width = f'%-{max(10, len(self.k) * 7 - 1)}s'
        for i in range(len(self.metrics_logger[self.metrics[0]])):
            epochs_string.append(width % f'{(i + 1) * self.evaluate_every} epochs')
            at_string.append(width % ' '.join([f'@{i:<5}' for i in self.k]))
        progression = [f'Model {self.uid}', 'Full progression:', '  '.join(epochs_string), '  '.join(at_string)]
        for k, v in self.metrics_logger.items():
            progression.append(f'{k:11}' + '  '.join([width % ' '.join([f'{g:.4f}' for g in j]) for j in v]))
        with open(self.progression_path, 'w') as f:
            f.write('\n'.join(progression))