from abc import ABC, abstractmethod

import numpy as np


class MetricsTracker(ABC):
    def __init__(self, logger, patience=5):
        self.logger = logger
        self.patience = patience
        self.epochs_no_improve = 0

    def update(self, results):
        for metric in results:
            self.metrics[metric].append(results[metric])
        self._check_best_metrics(results)

    @property
    def last_result(self):
        return {metric: values[-1] for metric, values in self.metrics.items() if values}

    def should_stop(self):
        return self.epochs_no_improve >= self.patience

    def print_best_results(self):
        self.logger.info("Best results:")
        self.log(self.best_metrics)

    def log(self, results=None):
        ''' save metrics from results in the logger '''
        for i in self._report(results).split('\n'):
            self.logger.info(i)

    def this_epoch_best(self):
        return self.metrics[self.main_metric][-1] == self.best_metrics[self.main_metric]

    def _check_best_metrics(self, result):
        if self._is_better(result[self.main_metric], self.best_metrics[self.main_metric]):
            self.best_metrics = result
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

    @abstractmethod
    def _is_better(self, a, b):
        ''' return True if a metric value is better than b '''

    @abstractmethod
    def _report(self, results=None):
        ''' return a string representing the metrics from results '''


class RankingMetricsTracker(MetricsTracker):

    def __init__(self, logger, k, *args, **kwargs):
        super().__init__(logger, *args, **kwargs)
        self.ks = sorted(k)
        self.metric_names = ['recall', 'precision', 'hit', 'f1', 'ndcg']
        self.main_metric = f"recall@{self.ks[0]}"
        self.best_metrics = {self.main_metric: -np.inf}
        self.metrics = {f"{metric}@{k}": [] for metric in self.metric_names for k in self.ks}

    def _is_better(self, a, b):
        return a > b

    def _report(self, results=None):
        ''' return a mostring with metrics for each k '''
        if results is None:
            results = self.last_result
        rows = ["           " + " ".join([f"@{k:<6}" for k in self.ks])]
        for metric in self.metric_names:
            rows.append(f'{metric:11}' + ' '.join([f'{results[f"{metric}@{k}"]:.4f}' for k in self.ks]))
        return "\n".join(rows)


class ScoringMetricsTracker(MetricsTracker):

    def __init__(self, logger, *args, **kwargs):
        super().__init__(logger, *args, **kwargs)
        self.main_metric = "Valid MSE"
        self.best_metrics = {self.main_metric: np.inf}
        self.metrics = {"Valid MSE": [], "Valid MAE": [], "Test  MSE": [], "Test  MAE": []}

    def _is_better(self, a, b):
        return a < b

    def _report(self, results=None):
        if results is None:
            results = {k: v for k, v in self.last_result.items() if 'Valid' in k}
        return "\n".join([f"{metric}: {values:.4f}" for metric, values in results.items()])
