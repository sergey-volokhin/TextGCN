import cProfile
import logging
import os
import pstats

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def hit(row):
    return row['intersecting_items'].shape[0] > 0


def recall(row):
    return row['intersecting_items'].shape[0] / row['y_true'].shape[0]


def precision(row, k):
    return row['intersecting_items'].shape[0] / k


def dcg(rel):
    return np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))


def ndcg(row, k):
    idcg = dcg(np.concatenate([np.ones(min(k, row['y_true'].shape[0])),
                               np.zeros(max(0, k - row['y_true'].shape[0]))]))
    rel = np.zeros(k)
    rel[np.where(np.isin(row['y_pred'][:k], row['intersecting_items']))] = 1
    return dcg(rel) / idcg


def get_logger(args):
    if args.quiet:
        args.logging_level = 'error'
    args.logging_level = {'debug': 10, 'info': 20, 'warn': 30, 'error': 40}[args.logging_level]
    logging.basicConfig(level=(logging.ERROR if args.quiet else args.logging_level),
                        format='%(asctime)-10s - %(levelname)s: %(message)s',
                        datefmt='%d/%m/%y %H:%M',
                        handlers=[logging.FileHandler(os.path.join(args.save_path, 'log.log')),
                                  logging.StreamHandler()])
    return logging.getLogger()


def early_stop(res):
    '''
        returns True if the difference between metrics
        from current and 2 previous epochs is less than 1e-4
    '''
    return len(res['recall']) > 2 and \
        all(np.allclose(m[-1], m[-2], atol=1e-4) for m in res.values()) and \
        all(np.allclose(m[-1], m[-3], atol=1e-4) for m in res.values())


def embed_text(sentences, name, path, bert_model, batch_size, device, logger):
    logger.info(f'Getting {name} embeddings')

    if os.path.exists(path):
        return torch.load(path)

    sentences_to_embed = sentences.unique().tolist()
    tokenization = tokenize_text(sentences_to_embed, bert_model, batch_size)
    bert = torch.nn.DataParallel(AutoModel.from_pretrained(bert_model)).to(device)
    embs = []
    with torch.no_grad():
        for batch in tqdm(tokenization, desc='embedding', dynamic_ncols=True):
            embs.append(bert(**batch).last_hidden_state[:, 0].detach().cpu().numpy())
    embeddings = np.concatenate(embs)
    mapping = {i: emb.tolist() for i, emb in zip(sentences_to_embed, embeddings)}
    result = sentences.map(mapping)
    logger.info('Saving calculated embeddings')
    torch.save(result, path)
    return result


def tokenize_text(sentences, bert_model, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(bert_model, strip_accents=True)
    token_batches = [sentences[j:j + batch_size] for j in range(0, len(sentences), batch_size)]

    if len(token_batches[-1]) == 0:
        token_batches = token_batches[:-1]
    tokenization = []
    for batch in tqdm(token_batches, desc='tokenization', dynamic_ncols=True):
        tokenization.append(tokenizer(batch,
                                      return_tensors="pt",
                                      padding=True,
                                      truncation=True,
                                      max_length=512))
    return tokenization


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


def sent_trans_embed_text(sentences, path, bert_model, batch_size, device, logger):
    ''' calculate SentenceBERT embeddings'''
    logger.info('Getting embeddings')

    if os.path.exists(path):
        return torch.load(path)

    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = SentenceTransformer(bert_model, device=device)
    sentences_to_embed = dedup_and_sort(sentences)

    embedding = model.encode(sentences_to_embed, batch_size=batch_size, convert_to_tensor=True)
    result = {i: j for i, j in zip(sentences_to_embed, embedding)}
    torch.save(result, path)

    return result
