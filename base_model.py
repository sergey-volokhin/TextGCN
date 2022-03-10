import os

import numpy as np
import pandas as pd
import torch
import torch.optim as opt
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from utils import early_stop, hit, minibatch, ndcg, precision, recall


class BaseModel(torch.nn.Module):
    ''' meta class that has model-agnostic utility functions '''

    def __init__(self, args, dataset):
        super().__init__()

        self._copy_args(args)
        self._copy_dataset_args(dataset)
        self._build_model()
        self.load_model()
        self.to(self.device)
        self._build_optimizer()

        self.current_epoch = 0
        self.metrics = ['recall', 'precision', 'hit', 'ndcg']
        self.metrics_logger = {i: np.zeros((0, len(args.k))) for i in self.metrics}
        self.w = SummaryWriter(self.save_path)

        self._save_code()
        self.logger.info(args)
        self.logger.info(self)

    def _copy_args(self, args):
        self.k = args.k
        self.lr = args.lr
        self.uid = args.uid
        self.quiet = args.quiet
        self.epochs = args.epochs
        self.logger = args.logger
        self.device = args.device
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.keep_prob = args.keep_prob
        self.save_path = args.save_path
        self.load_path = args.load_path
        self.layer_size = args.layer_size
        self.save_model = args.save_model
        self.batch_size = args.batch_size
        self.reg_lambda = args.reg_lambda
        self.evaluate_every = args.evaluate_every

    def _copy_dataset_args(self, dataset):
        self.graph = dataset.graph            # dgl graph
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.sampler = dataset.sampler
        self.n_batches = dataset.n_batches
        self.test_user_dict = dataset.test_user_dict
        self.embedding_user = dataset.embedding_user
        self.embedding_item = dataset.embedding_item
        self.get_user_pos_items = dataset.get_user_pos_items
        self._dropout_norm_matrix = dataset._dropout_norm_matrix
        self.adj = self.graph.adj(etype='bought', ctx=self.device)

    def _build_model(self):
        '''
            adding all the torch variables so we could
            move to cuda before making optimizers
        '''
        pass

    def _build_optimizer(self):
        self.optimizer = opt.Adam(self.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            verbose=(not self.quiet),
                                                            patience=5,
                                                            min_lr=1e-6)

    def get_loss(self, users, pos, neg):

        ''' normal loss '''
        # TODO change to only return representation of (users,pos,neg)
        users_emb, item_emb = self.get_representation()
        users_emb = users_emb[users]
        pos_emb = item_emb[pos]
        neg_emb = item_emb[neg]
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # inner product
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)  # inner product
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))  # bpr loss

        ''' regularization loss '''
        user_vec = self.embedding_user(users)
        pos_vec = self.embedding_item(pos)
        neg_vec = self.embedding_item(neg)
        reg_loss = (user_vec.norm(2).pow(2) + pos_vec.norm(2).pow(2) + neg_vec.norm(2).pow(2)) / len(users) / 2

        return loss + self.reg_lambda * reg_loss

    def step(self):
        '''
            one iteration over n_train number of samples
            i.e. one step of the training cycle
            returns total loss over all users
        '''
        total_loss = 0
        for arguments in tqdm(self.sampler(),
                              total=self.n_batches,
                              desc='current batch',
                              leave=False,
                              dynamic_ncols=True):
            self.optimizer.zero_grad()
            loss = self.get_loss(*arguments)
            total_loss += loss
            loss.backward()
            self.optimizer.step()

        total_loss = total_loss.item()
        self.w.add_scalar('Training_loss', total_loss, self.current_epoch)
        assert not np.isnan(total_loss), 'loss is nan while training'
        self.scheduler.step(total_loss)
        return total_loss

    def workout(self):
        ''' training loop called 'workout' because torch models already have internal .train method '''
        for epoch in trange(1, self.epochs + 1, desc='epochs', dynamic_ncols=True):
            self.train()
            self.current_epoch += 1
            loss = self.step()

            if epoch % self.evaluate_every:
                continue
            self.logger.info(f'Epoch {epoch}: loss = {loss}')

            self.evaluate()
            if self.save_model:
                self.checkpoint()
            if early_stop(self.metrics_logger):
                self.logger.warning(f'Early stopping triggerred at epoch {epoch}')
                break
        if self.save_model:
            self.checkpoint()
        self.w.flush()

    def evaluate(self):
        ''' calculate and report metrics for test users against predictions '''
        self.eval()
        predictions = self.predict()
        results = {i: np.zeros(len(self.k)) for i in self.metrics}
        for col in predictions.columns:
            predictions[col] = predictions[col].apply(np.array)
        for k in self.k:
            predictions[f'intersection_{k}'] = predictions.apply(lambda row: np.intersect1d(row['y_pred'][:k], row['y_true']), axis=1)

        ''' get metrics per user '''
        n_users = len(self.test_user_dict)
        for row in predictions.itertuples(index=False):
            r = self.test_one_user(row)
            for metric in r:
                results[metric] += r[metric]
        for metric in results:
            results[metric] /= n_users
            self.w.add_scalars(f'Test/{metric}',
                               {str(self.k[i]): results[metric][i] for i in range(len(self.k))},
                               self.current_epoch)

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
            users_emb, items_emb = self.get_representation()
            for batch_users in tqdm(minibatch(users, batch_size=self.batch_size),
                                    total=(len(users) - 1) // self.batch_size + 1,
                                    desc='test batches',
                                    leave=False,
                                    dynamic_ncols=True):

                # get the estimated user-item scores with matmul embedding matrices
                batch_user_emb = users_emb[torch.Tensor(batch_users).long().to(self.device)]
                rating = torch.matmul(batch_user_emb, items_emb.t())

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
        return result

    def _save_code(self):
        ''' saving the code to the folder with the model (for debugging) '''
        folder = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f'{self.save_path}/code', exist_ok=True)
        for file in os.listdir(folder):
            if file.endswith('.py'):
                os.system(f'cp {folder}/{file} {self.save_path}/code/{file}_')

    def load_model(self):
        ''' load torch weights from file '''
        if self.load_path:
            self.load_state_dict(torch.load(self.load_path))
            self.logger.info(f'Loaded model {self.load_path}')
            index = max([0] + [int(i[12:-4]) for i in os.listdir(self.save_path) if i.startswith('progression_')])
        else:
            self.logger.info(f'Created model {self.uid}')
            index = 0
        self.progression_path = f'{self.save_path}/progression_{index + 1}.txt'

    def checkpoint(self):
        ''' save current model and update the best one '''
        os.system(f'rm -f {self.save_path}/model_checkpoint*')
        torch.save(self.state_dict(), f'{self.save_path}/model_checkpoint{self.current_epoch}')
        if self.metrics_logger[self.metrics[0]][:, 0].max() == self.metrics_logger[self.metrics[0]][-1][0]:
            self.logger.info(f'Updating best model at epoch {self.current_epoch}')
            torch.save(self.state_dict(), f'{self.save_path}/model_best')

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

    def get_representation(self):
        '''
            get the users' and items' final representations using
            calculated embeddings and propagated through layers
        '''
        raise NotImplementedError
