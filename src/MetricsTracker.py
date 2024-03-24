import logging
from abc import ABC, abstractmethod

import numpy as np


class MetricsTracker(ABC):
    def __init__(self, logger, patience=5):
        self.logger = logger
        self.patience = patience
        self.epochs_no_improve = 0
        self.metrics = {}
        self.best_metrics = {}

    def __iadd__(self, results):
        ''' appends new results to self.metrics '''
        for metric in results:
            self.metrics[metric].append(results[metric])
        self._update_best_metrics(results)
        return self

    def __isub__(self, results):
        ''' removes last results from self.metrics '''
        for metric in results:
            self.metrics[metric].pop()
        return self

    @property
    def last_result(self):
        return {metric: values[-1] for metric, values in self.metrics.items() if values}

    def should_stop(self):
        return self.epochs_no_improve >= self.patience

    def print_best_results(self):
        self.logger.warn("Best results:")
        self.log(self.best_metrics, level='warn')

    def log(self, results=None, level='info'):
        ''' write metrics in the logger '''
        for i in self._report(results).split('\n'):
            self.logger.log(msg=i, level=logging._nameToLevel[level.upper()])

    def last_epoch_best(self):
        ''' check if results from the last epoch are the best '''
        return self.metrics[self.main_metric][-1] == self.best_metrics[self.main_metric]

    def _update_best_metrics(self, result):
        '''
        update best metrics if:
            1. current result is better than best metrics
            2. or best metrics don't exist yet
        '''
        if not self.best_metrics or self._is_better_than_cur_best(result):
            self.best_metrics.update(result)
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

    def _is_better_than_cur_best(self, result):
        ''' check if result is better than best metrics '''
        return self._metric_is_better(result[self.main_metric], self.best_metrics[self.main_metric])

    @abstractmethod
    def _metric_is_better(self, a, b):
        '''
        return True if 'a' metric value is better than 'b'
            i.e. "higher is better" or "lower is better"
        '''

    @abstractmethod
    def _report(self, results):
        ''' return a string representing the metrics from results '''


class RankingMetricsTracker(MetricsTracker):

    def __init__(self, logger, k, *args, **kwargs):
        super().__init__(logger, *args, **kwargs)
        self.ks = sorted(k)
        self.metric_names = ['recall', 'precision', 'hit', 'f1', 'ndcg']
        self.main_metric = f"recall@{self.ks[0]}"
        self.metrics = {f"{metric}@{k}": [] for metric in self.metric_names for k in self.ks}
        self.best_metrics = {m: -np.inf for m in self.metrics}

    @staticmethod
    def _metric_is_better(a, b):
        return a > b

    def _report(self, results=None):
        if results is None:
            results = self.last_result
        rows = ["           " + " ".join([f"@{k:<6}" for k in self.ks])]
        for metric in self.metric_names:
            rows.append(f'{metric:11}' + ' '.join([f'{results[f"{metric}@{k}"]:.4f}' for k in self.ks]))
        return "\n".join(rows)


class ScoringMetricsTracker(MetricsTracker):

    def __init__(self, logger, *args, **kwargs):
        super().__init__(logger, *args, **kwargs)
        self.main_metric = "valid_mse"
        # self.main_metric = "valid_mae"
        self.metrics = {'train_mse': [], 'train_mae': [], "valid_mse": [], "valid_mae": [], "test_mse": [], "test_mae": []}
        self.best_metrics = {m: np.inf for m in self.metrics}

    @staticmethod
    def _metric_is_better(a, b):
        return a < b

    def _report(self, results=None):
        if results is None:
            results = {k: v for k, v in self.last_result.items() if 'test' not in k}
        return "\n".join([f"{metric:9} {values:.4f}" for metric, values in results.items()])
