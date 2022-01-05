import logging

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def get_logger(args):
    logging.basicConfig(level=(logging.ERROR if args.quiet else logging.INFO),
                        format='%(asctime)-10s - %(levelname)s: %(message)s',
                        datefmt='%d/%m/%y %H:%M',
                        handlers=[logging.FileHandler(args.save_path + 'log.log'), logging.StreamHandler()])
    return logging.getLogger()


def early_stop(res):
    return len(res['recall']) > 1 and all(np.allclose(m[-1], m[-2], atol=1e-4) for m in res.values())


def report_best_scores(model):
    ''' report best scores '''
    model.metrics_logger = {k: np.array(v) for k, v in model.metrics_logger.items()}
    best_rec_0 = max(model.metrics_logger['recall'][:, 0])
    idx = list(model.metrics_logger['recall'][:, 0]).index(best_rec_0)
    model.logger.info(f'Best Metrics (at epoch {idx * model.evaluate_every}):')
    for k, v in model.metrics_logger.items():
        model.logger.info(f'{k} {" "*(9-len(k))} {v[idx][0]:.4f} {v[idx][-1]:.4f}')
    model.logger.info(f'Best model is saved in `{model.save_path}model_best`')
    model.logger.info(f'Full progression of metrics is saved in `{model.progression_path}`')


def init_bert(args):
    torch.cuda.empty_cache()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, strip_accents=True)
    if torch.cuda.is_available() and args.gpu:
        batch_size = args.batch_size
        model = torch.nn.DataParallel(BertModel.from_pretrained(args.bert_model), device_ids=[int(x) for x in args.gpu.split(', ')])
        model.to('cuda:' + args.gpu.split(',')[0])
    else:
        batch_size = 16
        model = BertModel.from_pretrained(args.bert_model)
    return tokenizer, model, batch_size


def embed_text(sentences, tokenizer, model, batch_size):
    num_samples = len(sentences)
    token_batches = [sentences[j * batch_size:(j + 1) * batch_size] for j in range(num_samples // batch_size)] + \
        [sentences[(num_samples // batch_size) * batch_size:]]
    embed_batches = []
    for batch in tqdm(token_batches, desc='tokenization', dynamic_ncols=True):
        embed_batches.append(tokenizer(batch,
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=True,
                                       max_length=512))
    with torch.no_grad():
        outputs = torch.cat([model(**batch).pooler_output for batch in tqdm(embed_batches,
                            desc='embedding', dynamic_ncols=True)])
    return outputs


def draw_bipartite(graph):
    nx_g = dgl.to_homogeneous(graph).to_networkx().to_undirected()
    pos = nx.drawing.layout.bipartite_layout(nx_g, range(len(nx_g.nodes())//2))
    nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()
