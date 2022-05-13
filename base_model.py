from collections import defaultdict
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.optim as opt
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm, trange

from utils import early_stop, hit, ndcg, precision, recall


class BaseModel(nn.Module):
    '''
        meta class with model-agnostic utility functions
        also works as custom lgcn (vs 'lightgcn' with torch_geometric layer)
    '''

    def __init__(self, args, dataset):
        super().__init__()

        self._copy_args(args)
        self._copy_dataset_args(dataset)
        self._add_vars()
        self._init_embeddings(args.emb_size)
        self.to(args.device)

        self.load_model(args.load)
        self._save_code()
        self.logger.info(args)
        self.logger.info(self)

    def _copy_args(self, args):
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
        self.slurm = args.slurm or args.quiet
        if args.single:
            self.layer_combination = self.layer_combination_single

    def _copy_dataset_args(self, dataset):
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_batches = dataset.n_batches
        self.norm_matrix = dataset.norm_matrix
        self.train_user_dict = dataset.train_user_dict
        self.true_test_lil = dataset.true_test_lil
        self.test_batches = dataset.test_batches

    def _init_embeddings(self, emb_size):
        ''' randomly initialize entity embeddings '''
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users,
                                           embedding_dim=emb_size).to(self.device)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_items,
                                           embedding_dim=emb_size).to(self.device)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def _add_vars(self):
        ''' add remaining variables '''
        self.metrics = ['recall', 'precision', 'hit', 'ndcg', 'f1']
        self.metrics_logger = {i: np.zeros((0, len(self.k))) for i in self.metrics}
        self.w = SummaryWriter(self.save_path)
        self.training = False

    @property
    def _dropout_norm_matrix(self):
        '''
            drop elements from adj table
            to help with overfitting
        '''
        index = self.norm_matrix.indices().t()
        values = self.norm_matrix.values()
        random_index = (torch.rand(len(values)) + (1 - self.dropout)).int().bool()
        index = index[random_index]
        values = values[random_index] / (1 - self.dropout)
        matrix = torch.sparse.FloatTensor(index.t(), values, self.norm_matrix.size())
        return matrix.coalesce().to(self.device)

    def _get_optimizer(self):
        self.optimizer = opt.Adam(self.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            verbose=(not self.quiet),
                                                            patience=5,
                                                            min_lr=1e-8)

    @property
    def embedding_matrix(self):
        ''' get the embedding matrix of 0th layer '''
        return torch.cat([self.embedding_user.weight, self.embedding_item.weight])

    @property
    def representation(self):
        '''
            aggregate embeddings from neighbors for each layer,
            combine layers into users' and items' final representations
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

        self._get_optimizer()

        for epoch in trange(1, self.epochs + 1, desc='epochs', disable=self.quiet):
            self.train()
            self.training = True
            self._loss_values = defaultdict(float)
            total_loss = 0
            for data in tqdm(batches,
                             desc='batches',
                             leave=False,
                             dynamic_ncols=True,
                             disable=self.slurm):
                self.optimizer.zero_grad()
                loss = self.get_loss(*data.to(self.device).t())
                if loss.isnan():
                    self.logger.error(f'loss is NA at epoch {epoch}')
                    exit()
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            self.w.add_scalar('Training_loss', total_loss, epoch)
            self.scheduler.step(total_loss)

            if epoch % self.evaluate_every:
                continue

            self.logger.info(f"Epoch {epoch}: {' '.join([f'{k} = {v:.4f}' for k,v in self._loss_values.items()])}")

            self.evaluate(epoch)
            self.checkpoint(epoch)

            if early_stop(self.metrics_logger):
                self.logger.warning(f'Early stopping triggerred at epoch {epoch}')
                break
        else:
            self.checkpoint(self.epochs)
        self.logger.info(f'Full progression of metrics is saved in `{self.progression_path}`')

    def layer_aggregation(self, norm_matrix, emb_matrix):
        '''
            aggregate the neighbor's representations
            to get next layer node representation.

            default: normalized sum
        '''
        return torch.sparse.mm(norm_matrix, emb_matrix)

    def layer_combination(self, vectors):
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

    def score(self, users, items, users_emb, item_emb):
        '''
            calculating score per pair:
            users_emb.shape === item_emb.shape
        '''
        return torch.sum(torch.mul(users_emb, item_emb), dim=1)

    def bpr_loss(self, users, pos, negs):
        ''' Bayesian Personalized Ranking pairwise loss '''
        users_emb, item_emb = self.representation
        users_emb = users_emb[users]
        pos_scores = self.score(users, pos, users_emb, item_emb[pos])
        loss = 0
        for neg in negs:
            neg_scores = self.score(users, neg, users_emb, item_emb[neg])
            loss += torch.mean(F.softplus(neg_scores - pos_scores))
            # loss += torch.mean(F.selu(neg_scores - pos_scores))
        res = loss / len(negs)
        self._loss_values['bpr'] += res
        return res

    def reg_loss(self, users, pos, negs):
        ''' regularization L2 loss '''
        loss = self.embedding_user(users).norm(2).pow(2) + self.embedding_item(pos).norm(2).pow(2)
        for neg in negs:
            loss += self.embedding_item(neg).norm(2).pow(2) / len(negs)
        res = self.reg_lambda * loss / len(users) / 2
        self._loss_values['reg'] += res
        return res

    def get_loss(self, users, pos, *negs):
        ''' get total loss per batch of users '''
        return self.bpr_loss(users, pos, negs) + self.reg_loss(users, pos, negs)

    def evaluate(self, epoch):
        ''' calculate and report metrics for test users against predictions '''
        self.eval()
        self.training = False
        predictions = self.predict()
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
            self.w.add_scalars(f'Test/{metric}',
                               {str(self.k[i]): results[metric][i] for i in range(len(self.k))},
                               epoch)

        ''' show metrics in log '''
        self.logger.info(' ' * 11 + ''.join([f'@{i:<6}' for i in self.k]))
        for i in results:
            self.metrics_logger[i] = np.append(self.metrics_logger[i], [results[i]], axis=0)
            self.logger.info(f'{i:11}' + ' '.join([f'{j:.4f}' for j in results[i]]))
        self.save_progression()

        return results

    def predict(self, save=False):
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

                rating = torch.matmul(users_emb[batch_users], items_emb.t())

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

    def test_one_user(self, row):
        result = {i: [] for i in self.metrics}
        for k in self.k:
            k_row = {'intersecting_items': getattr(row, f'intersection_{k}'),
                     'y_pred': row.y_pred,
                     'y_true': row.y_true}
            result['recall'].append(recall(k_row))
            result['precision'].append(precision(k_row, k))
            result['hit'].append(hit(k_row))
            result['ndcg'].append(ndcg(k_row, k))
            numerator = result['recall'][-1] * result['precision'][-1] * 2
            denominator = result['recall'][-1] + result['precision'][-1]
            result['f1'].append(np.divide(numerator,
                                          denominator,
                                          out=np.zeros_like(numerator),
                                          where=denominator != 0))
        return result

    def _save_code(self):
        ''' saving the code to the folder with the model (for debugging) '''
        folder = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(self.save_path, 'code'), exist_ok=True)
        for file in os.listdir(folder):
            if file.endswith('.py'):
                shutil.copyfile(os.path.join(folder, file),
                                os.path.join(self.save_path, 'code', file + '_'))

    def load_model(self, load_path):
        ''' load torch weights from file '''
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=self.device))
            self.logger.info(f'Loaded model {load_path}')
        else:
            self.logger.info(f'Created model {self.uid}')
        self.progression_path = f'{self.save_path}/progression.txt'

    def checkpoint(self, epoch):
        ''' save current model and update the best one '''
        if not self.save:
            return
        torch.save(self.state_dict(), f'{self.save_path}/checkpoint.pkl')
        if self.metrics_logger[self.metrics[0]][:, 0].max() == self.metrics_logger[self.metrics[0]][-1][0]:
            self.logger.info(f'Updating best model at epoch {epoch}')
            os.system(f'cp {self.save_path}/checkpoint.pkl {self.save_path}/best.pkl')
            shutil.copyfile(os.path.join(self.save_path, 'checkpoint.pkl'),
                            os.path.join(self.save_path, 'best.pkl'))

    def save_progression(self):
        ''' save all scores in one file for clarity '''
        epochs_string, at_string = [' ' * 9], [' ' * 9]
        width = f'%-{max(10, len(self.k) * 7 - 1)}s'
        for i in range(len(self.metrics_logger[self.metrics[0]])):
            epochs_string.append(width % f'{(i + 1) * self.evaluate_every} epochs')
            at_string.append(width % ' '.join([f'@{i:<5}' for i in self.k]))
        progression = [f'Model {self.uid}', 'Full progression:', '  '.join(epochs_string), '  '.join(at_string)]
        for k, v in self.metrics_logger.items():
            progression.append(f'{k:11}' + '  '.join([width % ' '.join([f'{g:.4f}' for g in j]) for j in v]))
        open(self.progression_path, 'w').write('\n'.join(progression))
