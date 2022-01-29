import logging
import os

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, DebertaV2Tokenizer


def get_logger(args):
    level = {'debug': 10, 'info': 20, 'warn': 30, 'error': 40}[args.logging_level]
    logging.basicConfig(level=(logging.ERROR if args.quiet else level),
                        format='%(asctime)-10s - %(levelname)s: %(message)s',
                        datefmt='%d/%m/%y %H:%M',
                        handlers=[logging.FileHandler(f'{args.save_path}/log.log'), logging.StreamHandler()])
    return logging.getLogger()


def early_stop(res):
    return len(res['recall']) > 1 and all(np.allclose(m[-1], m[-2], atol=1e-4) for m in res.values())


def embed_text(sentences, path, bert_model, batch_size, device):
    tokenization = tokenize_text(sentences, path, bert_model, batch_size)
    bert = torch.nn.DataParallel(AutoModel.from_pretrained(bert_model)).to(device)
    with torch.no_grad():
        embeddings = torch.cat([bert(**batch).last_hidden_state[0][0] for batch in tqdm(tokenization, desc='embedding', dynamic_ncols=True)])
    del bert
    torch.cuda.empty_cache()
    torch.save(embeddings, f'{path}/embeddings_{bert_model.split("/")[1]}.txt')
    return embeddings


def tokenize_text(sentences, path, bert_model, batch_size):
    path = f'{path}/tokenization_{bert_model.split("/")[1]}.txt'
    if not os.path.exists(path):
        tokenizer = DebertaV2Tokenizer.from_pretrained(bert_model, strip_accents=True)
        num_samples = len(sentences)
        token_batches = [sentences[j * batch_size:(j + 1) * batch_size] for j in range(num_samples // batch_size)] + \
                        [sentences[(num_samples // batch_size) * batch_size:]]
        tokenization = []
        for batch in tqdm(token_batches, desc='tokenization', dynamic_ncols=True):
            tokenization.append(tokenizer(batch,
                                          return_tensors="pt",
                                          padding=True,
                                          truncation=True,
                                          max_length=512))
        del tokenizer
        torch.cuda.empty_cache()
        torch.save(tokenization, path)
    else:
        tokenization = torch.load(path)
    return tokenization


def draw_bipartite(graph):
    ''' draw a bipartite graph '''
    import networkx as nx
    nx_g = dgl.to_homogeneous(graph).to_networkx().to_undirected()
    pos = nx.drawing.layout.bipartite_layout(nx_g, range(len(nx_g.nodes()) // 2))
    nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()
