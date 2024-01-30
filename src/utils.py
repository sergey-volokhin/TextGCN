import tiktoken
import logging
import os
import pickle
import time

import numpy as np
import openai
import pandas as pd
import torch
from more_itertools import chunked
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from tqdm.auto import tqdm


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


def calculate_scoring_metrics(y_pred, y_true):
    return {'mse': F.mse_loss(y_pred, y_true), 'mae': F.l1_loss(y_pred, y_true)}


def get_logger(config):
    if config.quiet:
        config.logging_level = 'error'
    config.logging_level = logging._nameToLevel[config.logging_level.upper()]
    logging.basicConfig(
        level=(logging.ERROR if config.quiet else config.logging_level),
        format='%(asctime)-10s - %(levelname)s: %(message)s',
        datefmt='%d/%m/%y %H:%M',
        handlers=[logging.FileHandler(os.path.join(config.save_path, 'log.log'), mode='w'), logging.StreamHandler()],
    )
    return logging.getLogger()


def num_tokens_from_list(strings: list[str], encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = sum([len(encoding.encode(string)) for string in strings])
    return num_tokens


def embed_text(
    sentences: list[str],
    path: str,
    model_name: str,
    batch_size: int,
    device,
    dimensions: int = 256,
) -> torch.Tensor:
    ''' calculate text embeddings, save as dict into pkl '''

    if os.path.exists(path):
        mapping = pickle.load(open(path, 'rb'))
        return torch.tensor([mapping[i] for i in sentences], device=device)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if model_name.startswith('text-embedding-3'):
        client = openai.OpenAI()
        batches = list(chunked(sentences, 8000))
        lengths = [num_tokens_from_list(batch) for batch in batches]
        assert all(i < 5_000_000 for i in lengths), f'batches too large: {lengths}'
        embeddings = []
        for batch in tqdm(batches, 'Embedding openai batches'):
            embeddings += [i.embedding for i in client.embeddings.create(input=batch, model=model_name, dimensions=dimensions).data]
            time.sleep(1)
        result = torch.tensor(embeddings)
    elif model_name.startswith('all-'):
        model = SentenceTransformer(model_name, device=device)
        result = model.encode(sentences, batch_size=batch_size, convert_to_tensor=True)
    else:
        raise ValueError(f'Unknown encoder: {model_name}')

    pickle.dump(dict(zip(sentences, result.tolist())), open(path, 'wb'))
    return result.to(device)


def subtract_tensor_as_set(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    '''
    quickly subtracts elements of the second tensor
    from the first tensor as if they were sets.

    copied from stackoverflow. no clue how this works
    '''
    return t1[(t2.repeat(t1.shape[0], 1).T != t1).T.prod(1) == 1].type(torch.int64)


def train_test_split_stratified(df, column='user_id', train_size=0.8, seed=42):
    """
    Splits the DataFrame into train, test, and val s.t.
    each set contains every unique value from the column

    1. remove users with less than 3 entries
    2. choose train_size s.t. at least 2 elements remain unchosen
    3. split the remaining elements equally between test and val
    """
    train_dfs = []
    test_dfs = []
    val_dfs = []

    for _, group in tqdm(df.groupby(column)):
        group = group.sample(frac=1, random_state=seed)
        train_end = min(int(train_size * len(group)), len(group) - 2)
        test_size = (len(group) - train_end) // 2

        train_dfs.append(group.iloc[:train_end])
        val_dfs.append(group.iloc[train_end:train_end + test_size])
        test_dfs.append(group.iloc[train_end + test_size:])

    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)
