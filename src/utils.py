import logging
import os

import numpy as np
import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer


def hit(row):
    return (row['intersecting_len'] > 0).astype(int)


def recall(row):
    return row['intersecting_len'] / row['y_true_len']


def precision(row, k):
    return row['intersecting_len'] / k


def dcg(rel, k):
    return np.sum((2 ** rel - 1) / np.log2(np.arange(2, k + 2)), axis=1)


def ndcg(row, k):
    row['ones'] = row['y_true_len'].apply(lambda r: np.ones(min(r, k)))
    row['zeros'] = row['y_true_len'].apply(lambda r: np.zeros(max(0, k - r)))
    arr = np.apply_along_axis(np.concatenate, 1, row[['ones', 'zeros']].values)
    idcg = dcg(arr, k)
    rel = np.apply_along_axis(lambda x: np.isin(x[0], x[1]), 1, row[[f'y_pred_{k}', f'intersection_{k}']].values)
    return dcg(rel, k) / idcg


def calculate_ranking_metrics(df, ks):
    '''
    computes all metrics for predictions for all users
    returns a dict of dicts with metrics for each k:
        {k: {metric: value}}
    '''

    result = {}
    df['y_true_len'] = df['y_true'].apply(len)

    for col in df.columns:
        df[col] = df[col].apply(np.array)

    for k in sorted(ks):
        ''' calculate intersections of y_pred and y_test '''
        df[f'intersection_{k}'] = df.apply(lambda row: np.intersect1d(row['y_pred'][:k], row['y_true']), axis=1)
        df[f'y_pred_{k}'] = df['y_pred'].apply(lambda x: x[:k])
        df['intersecting_len'] = df[f'intersection_{k}'].apply(len)

        rec = recall(df)
        prec = precision(df, k)
        numerator = rec * prec * 2
        denominator = rec + prec

        result.update({
            f'recall@{k}': rec.mean(),
            f'precision@{k}': prec.mean(),
            f'hit@{k}': hit(df).mean(),
            f'ndcg@{k}': ndcg(df, k).mean(),
            f'f1@{k}': np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0,
            ).mean()
        })
    return result


def calculate_scoring_metrics(y_pred, y_true, split='Valid'):
    return {f'{split} MSE': F.mse_loss(y_pred, y_true), f'{split} MAE': F.l1_loss(y_pred, y_true)}


def get_logger(config):
    if config.quiet:
        config.logging_level = 'error'
    config.logging_level = {'debug': 10, 'info': 20, 'warn': 30, 'error': 40}[config.logging_level]
    logging.basicConfig(
        level=(logging.ERROR if config.quiet else config.logging_level),
        format='%(asctime)-10s - %(levelname)s: %(message)s',
        datefmt='%d/%m/%y %H:%M',
        handlers=[logging.FileHandler(os.path.join(config.save_path, 'log.log'), mode='w'), logging.StreamHandler()],
    )
    return logging.getLogger()


def embed_text(
    sentences,
    path: str,
    bert_model: str,
    batch_size: int,
    device,
) -> torch.Tensor:
    ''' calculate SentenceBERT embeddings '''

    if os.path.exists(path):
        return torch.load(path, map_location=device)

    def dedup_and_sort(line):  # sort by num tokens, split collisions by lexicographical order
        return sorted(line.unique().tolist(), key=lambda x: (len(x.split()), x), reverse=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = SentenceTransformer(bert_model, device=device)
    sentences_to_embed = dedup_and_sort(sentences)

    embeddings = model.encode(sentences_to_embed, batch_size=batch_size)
    del model

    mapping = {i: emb for i, emb in zip(sentences_to_embed, embeddings)}
    result = torch.from_numpy(np.stack(sentences.map(mapping).values)).to(device=device)
    torch.save(result, path)
    return result


def subtract_tensor_as_set(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    '''
    quickly subtracts elements of the second tensor
    from the first tensor as if they were sets.

    copied from stackoverflow. no clue how this works
    '''
    return t1[(t2.repeat(t1.shape[0], 1).T != t1).T.prod(1) == 1].type(torch.int64)
