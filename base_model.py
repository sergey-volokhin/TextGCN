import os

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm, trange

from utils import early_stop
from metrics import hit, recall, precision, ndcg


class BaseModel(nn.Module):
    ''' meta class that has model-agnostic utility functions '''

    def __init__(self, args, dataset):
        super().__init__()

        self._copy_args(args)
        self._copy_dataset_args(dataset)
        self.logger.info(args)
        self._save_code()

        self.current_epoch = 0
        self.metrics = ['recall', 'precision', 'hit', 'ndcg']
        self.metrics_logger = {i: np.zeros((0, len(args.k))) for i in self.metrics}
        self.w = SummaryWriter(self.save_path)

    def _copy_args(self, args):
        self.k = args.k
        self.lr = args.lr
        self.uid = args.uid
        self.quiet = args.quiet
        self.epochs = args.epochs
        self.logger = args.logger
        self.device = args.device
        self.save_path = args.save_path
        self.load_path = args.load_path
        self.save_model = args.save_model
        self.evaluate_every = args.evaluate_every

    def _copy_dataset_args(self, dataset):
        self.graph = dataset.graph
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.test_user_dict = dataset.test_user_dict

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              verbose=(not self.quiet),
                                                              patience=5,
                                                              min_lr=1e-6)

    def step(self):
        '''
            one iteration over all train users
            i.e. one step of the training cycle
            returns total loss over all users
        '''
        total_loss = 0
        sampler, batches = self.get_sampler()
        for arguments in tqdm(sampler,
                              total=batches,
                              desc='current batch',
                              leave=False,
                              dynamic_ncols=True):
            self.optimizer.zero_grad()
            loss = self.get_loss(*arguments)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.cpu().item()
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
            torch.cuda.empty_cache()
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
        self.report_best_scores()
        self.w.close()

    def evaluate(self):
        ''' calculate and report metrics for test users against predictions '''
        self.eval()
        predictions = self.predict()
        results = {i: np.zeros(len(self.k)) for i in self.metrics}
        for col in predictions.columns:
            predictions[col] = predictions[col].apply(np.array)
        for k in self.k:
            predictions[f'intersection_{k}'] = predictions.apply(lambda row: np.intersect1d(row['y_pred'][:k], row['y_true']), axis=1)

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

        self.logger.info(' ' * 11 + ''.join([f'@{i:<6}' for i in self.k]))
        for i in results:
            self.metrics_logger[i] = np.append(self.metrics_logger[i], [results[i]], axis=0)
            self.logger.info(f'{i:11}' + ' '.join([f'{j:.4f}' for j in results[i]]))
        self.save_progression()

        return results

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
        for file in os.listdir(folder):
            if file.endswith('.py'):
                os.system(f'cp {folder}/{file} {self.save_path}')

    def load_model(self):
        ''' load torch weights from file '''
        if self.load_path:
            self.load_state_dict(torch.load(self.load_path))
            self.logger.info(f'Loaded model {self.load_path}')
        else:
            self.logger.info(f'Created model {self.uid}')
        index = max([0] + [int(i[12:-4]) for i in os.listdir(self.save_path) if i.startswith('progression_')])
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

    def report_best_scores(self):
        ''' print scores from best epoch on screen '''
        self.metrics_logger = {k: np.array(v) for k, v in self.metrics_logger.items()}
        idx = np.argmax(self.metrics_logger[self.metrics[0]][:, 0])
        self.logger.info(f'Best Metrics (at epoch {(idx + 1) * self.evaluate_every}):')
        self.logger.info(' ' * 11 + ' '.join([f'@{i:<5}' for i in self.k]))
        for k, v in self.metrics_logger.items():
            self.logger.info(f'{k:11}' + ' '.join([f'{j:.4f}' for j in v[idx]]))
        self.logger.info(f'Full progression of metrics is saved in `{self.progression_path}`')

    def get_loss(self, *args):
        ''' calculate loss given a batch of users and their positive and negative items '''
        raise NotImplementedError

    def get_sampler(self):
        '''
            returns a sampler and batch_size used in the `step` function
            sampler is iterated over and arguments are used to calculate loss in `get_loss` function:
            ```
            for args in sampler:
                loss = self.get_loss(*args)
            ```
        '''
        raise NotImplementedError

    def predict(self):
        '''
            returns a dataframe with predicted and true items for each test user:
            pd.DataFrame(columns=['y_pred', 'y_true'])
        '''
        raise NotImplementedError
