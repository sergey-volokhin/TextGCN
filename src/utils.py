import logging
import os
import pickle
import time
from functools import wraps

import numpy as np
import pandas as pd
import torch
from more_itertools import chunked
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, BitsAndBytesConfig, AutoTokenizer


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f'{func.__name__:<30} {time.perf_counter() - start:>6.2f} sec')
        return res
    return wrapper


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


def sort_process_unsort(func):
    ''' Decorator to sort the input by length, process it, and then unsort the output '''
    @wraps(func)
    def wrapper(sentences, **kwargs):
        sorted_strings_with_indices = sorted(enumerate(sentences), key=lambda x: -len(x[1]))
        sorted_strings = [string for _, string in sorted_strings_with_indices]
        processed_values = func(sentences=sorted_strings, **kwargs)
        unsorted_values = [None] * len(sentences)
        for (original_index, _), value in zip(sorted_strings_with_indices, processed_values):
            unsorted_values[original_index] = value
        if isinstance(processed_values, torch.Tensor):
            return torch.stack(unsorted_values)
        return unsorted_values
    return wrapper


@timeit
def embed_text(
    sentences: list[str],
    path: str,
    model_name: str,
    batch_size: int,
    logger,
    device,
    dimensions: int = 256,
) -> torch.Tensor:
    ''' calculate text embeddings, save as dict into pkl '''

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        mapping = pickle.load(open(path, 'rb'))
        to_embed = [i for i in sentences if i not in mapping]
        if not to_embed:
            logger.info('all sentences are embedded')
            return torch.tensor([mapping[i] for i in sentences], device=device)
    else:
        to_embed = sentences

    logger.info(f'embedding {len(to_embed)} sentences')
    if model_name.startswith('text-embedding-3'):
        result = embed_openai(to_embed, model_name, dimensions)
    elif model_name.startswith('all-'):  # or model_name in ['Salesforce/SFR-Embedding-Mistral']:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        result = model.encode(to_embed, batch_size=batch_size, convert_to_tensor=True)
    elif model_name in ['Salesforce/SFR-Embedding-Mistral']:
        result = embed_salesforce(to_embed, model_name=model_name, batch_size=batch_size)
    else:
        raise ValueError(f'Unknown encoder: {model_name}')

    new_result = dict(zip(to_embed, result.tolist()))

    if os.path.exists(path):
        new_result.update(mapping)
    pickle.dump(new_result, open(path, 'wb'))

    return torch.tensor([new_result[i] for i in sentences], device=device)


def embed_openai(sentences, model_name, dimensions):
    import openai
    import tiktoken

    def num_tokens_from_list(strings: list[str], encoding_name="cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = sum([len(encoding.encode(string)) for string in strings])
        return num_tokens

    client = openai.OpenAI()
    batches = list(chunked(sentences, 2000))
    lengths = [num_tokens_from_list(batch) for batch in batches]
    assert all(i < 5_000_000 for i in lengths), f'batches too large: {lengths}'
    embeddings = []
    for batch in tqdm(batches, 'Embedding openai batches'):
        embeddings += [i.embedding for i in client.embeddings.create(input=batch, model=model_name, dimensions=dimensions).data]
        time.sleep(1)
    return torch.tensor(embeddings)


@sort_process_unsort
@torch.no_grad()
def embed_salesforce(sentences, model_name, batch_size):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    all_embeddings = []
    batches = list(chunked(sentences, batch_size))
    for batch in tqdm(batches, desc="Salesforce batches"):
        batch_dict = tokenizer(batch, max_length=4096, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**batch_dict)
        batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        all_embeddings.append(batch_embeddings)
    return torch.cat(all_embeddings, dim=0)


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

    for _, group in tqdm(df.groupby(column), desc='reshuffle scoring data', dynamic_ncols=True):
        group = group.sample(frac=1, random_state=seed)
        train_end = min(int(train_size * len(group)), len(group) - 1)
        test_size = (len(group) - train_end) // 2  # add to test if only 1 element remains
        train_dfs.append(group.iloc[:train_end])
        val_dfs.append(group.iloc[train_end:train_end + test_size])
        test_dfs.append(group.iloc[train_end + test_size:])

    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)
