import numpy as np
import torch


def l2_loss_mean(x):
    ''' mean( sum(t ** 2) / 2) '''
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def hit(row):
    return row['intersecting_items'].shape[0] > 0


def recall(row):
    return row['intersecting_items'].shape[0] / row['y_test'].shape[0]


def precision(row):
    return row['intersecting_items'].shape[0] / row['y_pred'].shape[0]


def dcg(rel):
    return np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))


def ndcg(row, k):

    # ideal DCG when all relevant (test) items are first
    idcg = dcg(np.concatenate([np.ones(min(k, row['y_test'].shape[0],)),
                               np.zeros(max(0, k - row['y_test'].shape[0]))]))

    # real DCG
    rel = np.zeros(k)
    rel[np.where(np.isin(row['y_pred'][:k], row['intersecting_items']))] = 1
    numerator = dcg(rel)

    return numerator / idcg
