import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import DebertaV2Model, DebertaV2Tokenizer
import torch


def hit(row):
    return row['intersecting_items'].shape[0] > 0


def recall(row):
    return row['intersecting_items'].shape[0] / row['y_true'].shape[0]


def precision(row, k):
    return row['intersecting_items'].shape[0] / k


def dcg(rel):
    return np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))


def ndcg(row, k):
    idcg = dcg(np.concatenate([np.ones(min(k, row['y_true'].shape[0],)),
                               np.zeros(max(0, k - row['y_true'].shape[0]))]))
    rel = np.zeros(k)
    rel[np.where(np.isin(row['y_pred'][:k], row['intersecting_items']))] = 1
    numerator = dcg(rel)

    return numerator / idcg


def get_logger(args):
    if args.quiet:
        args.logging_level = 'error'
    args.logging_level = {'debug': 10, 'info': 20, 'warn': 30, 'error': 40}[args.logging_level]
    logging.basicConfig(level=(logging.ERROR if args.quiet else args.logging_level),
                        format='%(asctime)-10s - %(levelname)s: %(message)s',
                        datefmt='%d/%m/%y %H:%M',
                        handlers=[logging.FileHandler(f'{args.save_path}/log.log'), logging.StreamHandler()])
    return logging.getLogger()


def early_stop(res):
    '''
        returns True if the difference between metrics
        from current and previous epochs is less than 1e-4
    '''
    return len(res['recall']) > 1 and all(np.allclose(m[-1], m[-2], atol=1e-4) for m in res.values())


def minibatch(*tensors, **kwargs):
    batch_size = kwargs['batch_size']
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays):
    assert len(set(len(x) for x in arrays)) == 1, 'All inputs to shuffle must have the same length.'
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    return tuple(x[shuffle_indices] for x in arrays)


def embed_text(sentences, path, bert_model, batch_size, device):
    sentences_to_embed = sentences.unique().tolist()
    tokenization = tokenize_text(sentences_to_embed, path, bert_model, batch_size)
    bert = torch.nn.DataParallel(DebertaV2Model.from_pretrained(bert_model)).to(device)
    with torch.no_grad():
        embs = []
        for batch in tqdm(tokenization, desc='embedding', dynamic_ncols=True):
            embs.append(bert(**batch).last_hidden_state[:, 0].detach().cpu().numpy())
        embeddings = np.concatenate(embs)
    del bert
    torch.cuda.empty_cache()
    os.system(f'rm -f {path}/tokenization_{bert_model.split("/")[-1]}.torch')
    mapping = {i: emb.tolist() for i, emb in zip(sentences_to_embed, embeddings)}
    result = sentences.map(mapping)
    torch.save(result, f'{path}/embeddings_{bert_model.split("/")[-1]}.torch')
    return result


def tokenize_text(sentences, path, bert_model, batch_size):
    tok_path = f'{path}/tokenization_{bert_model.split("/")[-1]}.torch'
    if not os.path.exists(tok_path):
        tokenizer = DebertaV2Tokenizer.from_pretrained(bert_model, strip_accents=True)
        num_samples = len(sentences)
        token_batches = [sentences[j * batch_size:(j + 1) * batch_size] for j in range(num_samples // batch_size)] + \
                        [sentences[(num_samples // batch_size) * batch_size:]]
        if len(token_batches[-1]) == 0:
            token_batches = token_batches[:-1]
        tokenization = []
        for batch in tqdm(token_batches, desc='tokenization', dynamic_ncols=True):
            tokenization.append(tokenizer(batch,
                                          return_tensors="pt",
                                          padding=True,
                                          truncation=True,
                                          max_length=512))
        del tokenizer
        torch.cuda.empty_cache()
        torch.save(tokenization, tok_path)
    else:
        tokenization = torch.load(tok_path)
    return tokenization


def draw_bipartite(graph):
    ''' draw a bipartite graph '''
    import networkx as nx
    import matplotlib.pyplot as plt
    import dgl

    nx_g = dgl.to_homogeneous(graph).to_networkx().to_undirected()
    pos = nx.drawing.layout.bipartite_layout(nx_g, range(len(nx_g.nodes()) // 2))
    nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()
