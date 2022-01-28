import json
import logging
import os

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def get_logger(args):
    level = {'debug': 10, 'info': 20, 'warn': 30, 'error': 40}[args.logging_level]
    logging.basicConfig(level=(logging.ERROR if args.quiet else level),
                        format='%(asctime)-10s - %(levelname)s: %(message)s',
                        datefmt='%d/%m/%y %H:%M',
                        handlers=[logging.FileHandler(f'{args.save_path}/log.log'), logging.StreamHandler()])
    return logging.getLogger()


def early_stop(res):
    return len(res['recall']) > 1 and all(np.allclose(m[-1], m[-2], atol=1e-4) for m in res.values())


def init_bert(args):
    torch.cuda.empty_cache()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, strip_accents=True)
    if torch.cuda.is_available() and args.gpu:
        batch_size = args.batch_size
        model = torch.nn.DataParallel(BertModel.from_pretrained(args.bert_model)).to(args.device)
    else:
        batch_size = 16
        model = BertModel.from_pretrained(args.bert_model)
    return tokenizer, model, batch_size


def embed_text(sentences, device, tokenizer, model, batch_size):
    num_samples = len(sentences)
    token_batches = [sentences[j * batch_size:(j + 1) * batch_size] for j in range(num_samples // batch_size)] + \
                    [sentences[(num_samples // batch_size) * batch_size:]]
    embed_batches = []
    for batch in tqdm(token_batches, desc='tokenization', dynamic_ncols=True):
        embed_batches.append(tokenizer(batch,
                                    #    return_tensors="pt",
                                       padding=True,
                                       truncation=True,
                                       max_length=512).to(device))
    json.dump([dict(i) for i in embed_batches], open('tokenization.txt', 'w'))
    del tokenizer
    embed_batches = [{i: torch.Tensor(j) for i, j in z.items()} for z in embed_batches]
    torch.cuda.empty_cache()
    with torch.no_grad():
        outputs = torch.cat([model(**batch).pooler_output for batch in tqdm(embed_batches, desc='embedding', dynamic_ncols=True)])
    return outputs


def draw_bipartite(graph):
    ''' draw a bipartite graph '''
    import networkx as nx
    nx_g = dgl.to_homogeneous(graph).to_networkx().to_undirected()
    pos = nx.drawing.layout.bipartite_layout(nx_g, range(len(nx_g.nodes()) // 2))
    nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()
