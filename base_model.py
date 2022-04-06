import os

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm, trange

from utils import early_stop, hit, ndcg, precision, recall


class BaseModel(nn.Module):
    ''' meta class that has model-agnostic utility functions '''

    def __init__(self, args, dataset):
        super().__init__()

        self._copy_args(args)
        self._copy_dataset_args(dataset)
        self._init_embeddings(args.emb_size)
        self._add_vars()

        self.load_model(args.load)
        self._save_code()
        self.logger.info(args)
        self.logger.info(self)

    def _copy_args(self, args):
        self.k = args.k
        self.uid = args.uid
        self.epochs = args.epochs
        self.logger = args.logger
        self.device = args.device
        self.dropout = args.dropout
        self.n_layers = args.n_layers
        self.save_path = args.save_path
        self.save_model = args.save_model
        self.batch_size = args.batch_size
        self.reg_lambda = args.reg_lambda
        self.evaluate_every = args.evaluate_every
        self.slurm = args.slurm

    def _copy_dataset_args(self, dataset):
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_batches = dataset.n_batches
        self.norm_matrix = dataset.norm_matrix
        self.test_user_dict = dataset.test_user_dict
        self.get_user_pos_items = dataset.get_user_pos_items

    def _init_embeddings(self, emb_size):
        ''' randomly initialize entity embeddings '''
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users,
                                           embedding_dim=emb_size).to(self.device)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_items,
                                           embedding_dim=emb_size).to(self.device)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()

    def _add_vars(self):
        ''' adding all the remaining variables '''
        self.metrics = ['recall', 'precision', 'hit', 'ndcg', 'f1']
        self.metrics_logger = {i: np.zeros((0, len(self.k))) for i in self.metrics}
        self.w = SummaryWriter(self.save_path)

    def get_loss(self, users, pos, neg):

        ''' normal loss '''
        # TODO change to only return representation of (users,pos,neg)
        users_emb, item_emb = self.representation
        users_emb = users_emb[users]
        pos_emb = item_emb[pos]
        neg_emb = item_emb[neg]
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        ''' regularization loss '''
        user_vec = self.embedding_user(users)
        pos_vec = self.embedding_item(pos)
        neg_vec = self.embedding_item(neg)
        reg_loss = (user_vec.norm(2).pow(2) + pos_vec.norm(2).pow(2) + neg_vec.norm(2).pow(2)) / len(users) / 2

        return loss + self.reg_lambda * reg_loss

    def evaluate(self, epoch):
        ''' calculate and report metrics for test users against predictions '''
        self.eval()
        predictions = self.predict()
        results = {i: np.zeros(len(self.k)) for i in self.metrics}
        for col in predictions.columns:
            predictions[col] = predictions[col].apply(np.array)
        for k in self.k:
            predictions[f'intersection_{k}'] = predictions.apply(
                lambda row: np.intersect1d(row['y_pred'][:k], row['y_true']), axis=1)

        ''' get metrics per user & aggregates '''
        n_users = len(self.test_user_dict)
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
        '''

        users = list(self.test_user_dict)
        y_pred, y_true = [], []
        with torch.no_grad():  # don't calculate gradient since we only predict
            users_emb, items_emb = self.representation

            batches = [users[j:j + self.batch_size] for j in range(0, len(users), self.batch_size)]
            batches = batches if self.slurm else tqdm(batches, desc='batches', leave=False, dynamic_ncols=True)

            for batch_users in batches:

                # get the estimated user-item scores with matmul embedding matrices
                batch_user_emb = users_emb[torch.Tensor(batch_users).long().to(self.device)]
                rating = self.f(torch.matmul(batch_user_emb, items_emb.t()))

                # set scores for train items to be -inf so we don't recommend them
                exclude_index, exclude_items = [], []
                for ind, items in enumerate(self.get_user_pos_items(batch_users)):
                    exclude_index += [ind] * len(items)
                    exclude_items.append(items)
                exclude_items = np.concatenate(exclude_items)
                rating[exclude_index, exclude_items] = np.NINF

                # select top-k items with highest ratings
                _, rank_indices = torch.topk(rating, k=max(self.k))
                y_pred += list(rank_indices.cpu().numpy().tolist())
                y_true += [self.test_user_dict[u] for u in batch_users]

        predictions = pd.DataFrame.from_dict({'y_pred': y_pred, 'y_true': y_true})
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
        os.makedirs(f'{self.save_path}/code', exist_ok=True)
        for file in os.listdir(folder):
            if file.endswith('.py'):
                os.system(f'cp {folder}/{file} {self.save_path}/code/{file}_')

    def load_model(self, load_path):
        ''' load torch weights from file '''
        if load_path:
            self.load_state_dict(torch.load(load_path))
            self.logger.info(f'Loaded model {load_path}')
        else:
            self.logger.info(f'Created model {self.uid}')
        self.progression_path = f'{self.save_path}/progression.txt'

    def checkpoint(self, epoch):
        ''' save current model and update the best one '''
        if not self.save_model:
            return
        torch.save(self.state_dict(), f'{self.save_path}/checkpoint.pkl')
        if self.metrics_logger[self.metrics[0]][:, 0].max() == self.metrics_logger[self.metrics[0]][-1][0]:
            self.logger.info(f'Updating best model at epoch {epoch}')
            os.system(f'cp {self.save_path}/checkpoint.pkl {self.save_path}/best.pkl')

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
        self.logger.info(f'Full progression of metrics is saved in `{self.progression_path}`')

    @property
    def representation(self):
        '''
            get the users' and items' final representations
            propagated through all the layers
        '''
        curent_lvl_emb_matrix = self.embedding_matrix()
        norm_matrix = self._dropout_norm_matrix
        node_embed_cache = [curent_lvl_emb_matrix]
        for _ in range(self.n_layers):
            curent_lvl_emb_matrix = self.layer_propagate(norm_matrix, curent_lvl_emb_matrix)
            node_embed_cache.append(curent_lvl_emb_matrix)
        aggregated_embeddings = self.layer_aggregation(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def layer_propagate(self, *args, **kwargs):
        '''
            propagate the current layer embedding matrix
            through and get the matrix of the next layer
        '''
        raise NotImplementedError

    def layer_aggregation(self, *args, **kwargs):
        '''
            given embeddings from all layers
            combine them into final representation matrix
        '''
        raise NotImplementedError

    def embedding_matrix(self, *args, **kwargs):
        ''' get the embedding matrix of 0th layer '''
        raise NotImplementedError

    @property
    def _dropout_norm_matrix(self):
        '''
            drop self.dropout elements from adj table
            to help with overfitting
        '''
        index = self.norm_matrix.indices().t()
        values = self.norm_matrix.values()
        random_index = (torch.rand(len(values)) + (1 - self.dropout)).int().bool()
        index = index[random_index]
        values = values[random_index] / (1 - self.dropout)
        return torch.sparse.FloatTensor(index.t(), values, self.norm_matrix.size()).to(self.device)

    def forward(self, batches, optimizer, scheduler):
        ''' training function '''

        for epoch in trange(1, self.epochs + 1, desc='epochs'):
            self.train()
            total_loss = 0
            batches = batches if self.slurm else tqdm(batches, desc='batches', leave=False, dynamic_ncols=True)
            for data in batches:
                optimizer.zero_grad()
                loss = self.get_loss(*data.to(self.device).t())
                total_loss += loss
                loss.backward()
                optimizer.step()
            self.w.add_scalar('Training_loss', total_loss, epoch)
            scheduler.step(total_loss)

            if epoch % self.evaluate_every:
                continue

            self.logger.info(f'Epoch {epoch}: loss = {total_loss}')
            self.evaluate(epoch)
            self.checkpoint(epoch)

            if early_stop(self.metrics_logger):
                self.logger.warning(f'Early stopping triggerred at epoch {epoch}')
                break
        else:
            self.checkpoint(self.epochs)
