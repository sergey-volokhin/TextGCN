import os
import uuid
import warnings

import numpy as np
import torch
from tqdm import tqdm, trange

from dataloader import DataLoader
from model import Model
from parser import parse_args
from utils import get_logger, early_stop, report_best_scores, draw_bipartite


# def step(model):
#     sampler = model.dataset.sampler(batch_size=model.batch_size, pos_mode=model.sampler_mode)
#     bars = tqdm(sampler, desc='CF', dynamic_ncols=True, leave=False, total=model.num_batches)
#     loss_total = 0
#     for arguments in bars:
#         model.optimizer.zero_grad()
#         loss = model.get_loss(*arguments[:-1])
#         loss.backward()
#         model.optimizer.step()
#         loss_total += loss.item()
#     if np.isnan(loss_total):
#         model.logger.error('loss is nan.')
#         exit()
#     model.scheduler.step(loss_total)
#     return loss_total


# def train(model):
#     for epoch in trange(1, args.epochs + 1, desc='epochs', dynamic_ncols=True):
#         model.train()

#         with torch.no_grad():
#             model.graph.edata['w'] = model.compute_attention(model.graph)
#         loss = step(model)

#         if epoch % args.evaluate_every != 0:
#             continue
#         torch.cuda.empty_cache()
#         logger.info(f'Epoch {epoch}: loss = {loss}')
#         model.evaluate()
#         if args.save_model:
#             model.checkpoint(epoch)
#         if early_stop(model.metrics_logger):
#             logger.warning(f'Early stopping triggerred at epoch {epoch}')
#             break
#     if args.save_model:
#         model.checkpoint(epoch)
#     report_best_scores(model)


if __name__ == '__main__':

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.logger = logger = get_logger(args)

    logger.info(f'Model {args.uid}')
    logger.info(args)
    args.data_path = os.path.realpath(os.path.join(os.path.abspath(__file__), '../datasets'))

    dataset = DataLoader(args, seed=args.seed)
    model = Model(args, dataset)

    draw_bipartite(dataset.graph)

    # # saving the code version that is running to the folder with the model
    # for file in ['dataset.py', 'main.py', 'kgat.py', 'utils.py']:
    #     os.system(f'cp {file} {args.save_path}')

    # train(model)
    # if args.predict:
    #     model.predict()
