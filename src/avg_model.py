import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from surprise import Dataset, Reader, accuracy
from surprise.prediction_algorithms import SVD, BaselineOnly, SVDpp, KNNBaseline

warnings.filterwarnings("ignore")


def mse(df):
    return round(sk_mse(df['prediction'], df['rating']), 4)


def mae(df):
    return round(sk_mae(df['prediction'], df['rating']), 4)


def metric(df, series):
    df['prediction'] = df['user_id'].apply(lambda x: series[x])
    return {'MSE': mse(df), 'MAE': mae(df)}


class AvgScore:

    def __init__(self, data):
        self.train_df = pd.read_table(os.path.join(data, 'train_scoring.tsv'))
        self.valid_df = pd.read_table(os.path.join(data, 'valid_scoring.tsv'))
        self.test_df = pd.read_table(os.path.join(data, 'test_scoring.tsv'))

        self.n_users = self.train_df.user_id.nunique()
        self.n_items = self.train_df.asin.nunique()
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.n_val = self.valid_df.shape[0]
        # print(f"n_train:    {self.n_train:-7}")
        # print(f"n_val:      {self.n_val:-7}")
        # print(f"n_test:     {self.n_test:-7}")
        # print(f"n_users:    {self.n_users:-7}")
        # print(f"n_items:    {self.n_items:-7}")

        vc = pd.concat([self.train_df, self.valid_df, self.test_df]).rating.value_counts()
        # for score in vc.index:
        #     print(f'{score}: {vc[score]:>7}')

        data = pd.concat([self.train_df, self.valid_df, self.test_df])
        data = data[~data.user_id.str.startswith('gen_')]

        vc = data.asin.value_counts()
        self.tail_items = vc[vc < 5].index
        self.head_items = vc[vc > 4].index
        self.test_tail = self.test_df[self.test_df.asin.isin(self.tail_items)]
        self.test_head = self.test_df[self.test_df.asin.isin(self.head_items)]

        self.calculate_values()

        self.models = (
            ('u_mean', self.user_mean),
            ('median', self.user_median),
            # ('mode', self.user_mode),
            ('all_5', self.user_fives),
            ('avg', self.global_avg),
            ('all_4', self.user_fours),
        )

    def calculate_values(self):
        self.user_mean = self.train_df.groupby('user_id')['rating'].mean()
        self.user_median = self.train_df.groupby('user_id')['rating'].median()
        # self.user_mode = (
        #     self.train_df.groupby('user_id')['rating']
        #     .agg(pd.Series.mode)
        #     .apply(lambda x: sum(x) / len(x) if isinstance(x, np.ndarray) else x)
        # )
        self.user_fives = pd.Series(5, index=self.train_df['user_id'].unique())
        self.user_fours = pd.Series(4, index=self.train_df['user_id'].unique())
        self.global_avg = pd.Series(self.train_df['rating'].mean(), index=self.train_df['user_id'].unique())

    def predict(self):
        results = {'val': {}, 'test': {}}
        for name, series in self.models:
            results['val'][name] = metric(self.valid_df, series)
            results['test'][name] = metric(self.test_df, series)
        return results

    def predict_head_tail(self):
        results = {'head': {}, 'tail': {}}
        for name, series in self.models:
            results['tail'][name] = metric(self.test_tail, series)
            results['head'][name] = metric(self.test_head, series)
        return results

    def print_results(self, results):
        for split in results:
            for metric in ['MSE', 'MAE']:
                print(f'{split:4} {metric}  ' + ' '.join([f"{results[split][i][metric]:.4f}" for i in results[split]]))


def run_headtail(model, head, tail):
    for name, dataset in zip(['head', 'tail'], [head, tail]):
        preds = model.test(dataset.build_full_trainset().build_testset())
        print(f'{name:4} mse {accuracy.mse(preds, verbose=False):.4f}')
        print(f'{name:4} mae {accuracy.mae(preds, verbose=False):.4f}')


def read_headtail(train_df, valid_df, test_df):
    data = pd.concat([train_df, valid_df, test_df])
    data = data[~data.user_id.str.startswith('gen_')]
    vc = data.asin.value_counts()
    tail_items = vc[vc < 5].index
    head_items = vc[vc > 4].index
    test_tail = test_df[test_df.asin.isin(tail_items)]
    test_head = test_df[test_df.asin.isin(head_items)]
    return test_head, test_tail


def avg(path):
    model = AvgScore(path)
    results = model.predict()
    print(' ' * 10 + ' '.join([f'{i:<6}' for i in results[list(results.keys())[0]]]))
    model.print_results(results)
    results = model.predict_head_tail()
    model.print_results(results)
    print()


def surprise(path):
    setups = {
        BaselineOnly: {},
        KNNBaseline: {'k': 10, 'min_k': 10, 'sim_options': {'name': 'pearson', 'user_based': True}},
        SVD: {'n_epochs': 50, 'lr_all': 0.01, 'reg_all': 0.5, 'n_factors': 50},
        # SVDpp: {'n_epochs': 50, 'lr_all': 0.01, 'reg_all': 0.5, 'n_factors': 50},
    }

    train_df, valid_df, test_df = read_data(path)

    reader = Reader(rating_scale=(1, 5))
    train = Dataset.load_from_df(train_df, reader)
    test = Dataset.load_from_df(test_df, reader)
    valid = Dataset.load_from_df(valid_df, reader)

    # find_model(data)
    # gridsearch(train)

    models = {}
    for model_class, params in setups.items():
        models[model_class] = run_surprise(model_class, params, train, valid, test)

    # run evaluation for tail and head itesm separately
    test_head, test_tail = read_headtail(train_df, valid_df, test_df)
    head = Dataset.load_from_df(test_head, reader)
    tail = Dataset.load_from_df(test_tail, reader)
    for model_class in models:
        print(f'{model_class.__name__}')
        run_headtail(models[model_class], head, tail)


def run_surprise(model_class, params, train, valid, test):
    print('\t', model_class.__name__)
    model = model_class(**params, verbose=False)
    model.fit(train.build_full_trainset())
    for name, dataset in zip(['valid', 'test'], [valid, test]):
        preds = model.test(dataset.build_full_trainset().build_testset())
        print(f'{name:5} mse {accuracy.mse(preds, verbose=False):.4f}')
        print(f'{name:5} mae {accuracy.mae(preds, verbose=False):.4f}')
    return model


def read_data(path):
    train_df = pd.read_table(os.path.join(path, 'train_scoring.tsv'))
    valid_df = pd.read_table(os.path.join(path, 'valid_scoring.tsv'))
    test_df = pd.read_table(os.path.join(path, 'test_scoring.tsv'))
    columns = ['user_id', 'asin', 'rating']
    return train_df[columns], valid_df[columns], test_df[columns]


if __name__ == '__main__':
    try:
        path = sys.argv[1]
    except Exception:
        print('Usage: python avg_model.py <path>')
        exit()
    # avg(path)
    # surprise(path)

    path = f'{path}/reshuffle_{{}}'
    for seed in range(5):
        avg(path.format(seed))

    for seed in range(5):
        surprise(path.format(seed))
