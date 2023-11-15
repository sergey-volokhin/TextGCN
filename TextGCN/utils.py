import cProfile
import logging
import os
import pstats
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def hit(row):
    return (row['intersecting_len'] > 0).astype(int)


def recall(row):
    return row['intersecting_len'] / row['y_true_len']


def precision(row, k: int):
    return row['intersecting_len'] / k


def dcg(rel, k: int):
    return np.sum((2 ** rel - 1) / np.log2(np.arange(2, k + 2)), axis=1)


def ndcg(row, k):
    row['ones'] = row['y_true_len'].apply(lambda r: np.ones(min(r, k)))
    row['zeros'] = row['y_true_len'].apply(lambda r: np.zeros(max(0, k - r)))
    arr = np.apply_along_axis(np.concatenate, 1, row[['ones', 'zeros']].values)
    idcg = dcg(arr, k)
    rel = np.apply_along_axis(lambda x: np.isin(x[0], x[1]), 1, row[[f'y_pred_{k}', f'intersection_{k}']].values)
    return dcg(rel, k) / idcg


def get_logger(args):
    if args.quiet:
        args.logging_level = 'error'
    args.logging_level = {'debug': 10, 'info': 20, 'warn': 30, 'error': 40}[args.logging_level]
    logging.basicConfig(
        level=(logging.ERROR if args.quiet else args.logging_level),
        format='%(asctime)-10s - %(levelname)s: %(message)s',
        datefmt='%d/%m/%y %H:%M',
        handlers=[logging.FileHandler(os.path.join(args.save_path, 'log.log'), mode='w'), logging.StreamHandler()],
    )
    return logging.getLogger()


def early_stop(res: dict[str, list[float] | np.ndarray]) -> bool:
    '''
    returns True if:
     the difference between metrics from current and 2 previous epochs is less than 1e-4
     or the last 3 epochs are yielding strictly declining values for all metrics
    '''
    if len(res['recall']) < 3:
        return False
    declining = all(np.less(m[-1], m[-2]).all() and np.less(m[-2], m[-3]).all() for m in res.values())
    converged = all(np.allclose(m[-1], m[-2], atol=1e-4) for m in res.values()) and \
                all(np.allclose(m[-1], m[-3], atol=1e-4) for m in res.values())
    return converged or declining


def tokenize_text(
    sentences: list[str],
    bert_model: str,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained(bert_model, strip_accents=True)
    token_batches = [sentences[j:j + batch_size] for j in range(0, len(sentences), batch_size)]

    if len(token_batches[-1]) == 0:
        token_batches = token_batches[:-1]
    tokenization = []
    for batch in tqdm(token_batches, desc='tokenization', dynamic_ncols=True):
        tokenization.append(
            tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
        )
    return tokenization


def embed_text(
    sentences,
    path: str,
    bert_model: str,
    batch_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    ''' calculate SentenceBERT embeddings '''

    if os.path.exists(path):
        return torch.load(path, map_location=device)

    def dedup_and_sort(line):
        return sorted(line.unique().tolist(), key=lambda x: len(x.split(" ")), reverse=True)

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


def profile(func):
    ''' function profiler to monitor time it takes for each call '''

    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

    return wrapper
