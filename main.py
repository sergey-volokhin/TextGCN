import cProfile
import pstats

import numpy as np
import torch
import torch.optim as opt
from rich import pretty
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataloader import MyDataLoader as DataLoaderLightGCN
from lightgcn import (LightGCN, LightGCNAttn, LightGCNSingle, LightGCNWeight,
                      LightSingleGCNAttn)
from text_model import DataLoaderKG, TextModelKG
from text_model_reviews import NGCF, DataLoaderReviews, TextModelReviews
from parser import parse_args

pretty.install()


def main():
    dataset = loader_class(args)
    model = model_class(args, dataset)
    model()
    if args.predict:
        model.predict(save=True)


def parallel_main():
    dataset = loader_class(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = model_class(args, dataset)

    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(torch.device("cuda"))

    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   verbose=(not args.quiet),
                                                   patience=5,
                                                   min_lr=1e-6)
    for epoch in trange(1, args.epochs + 1, desc='epochs'):
        model.train()
        total_loss = 0
        for data in tqdm(loader, desc='data', leave=False):
            optimizer.zero_grad()
            loss = model.module.get_loss(*data.to(args.device).t())
            total_loss += loss
            loss.backward()
            optimizer.step()
        model.module.w.add_scalar('Training_loss', total_loss, epoch)
        scheduler.step(total_loss)

        if epoch % args.evaluate_every:
            continue

        model.module.logger.info(f'Epoch {epoch}: loss = {total_loss}')
        model.module.evaluate()
        model.module.checkpoint()


def profile(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
    return wrapper


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loader_class, model_class = {'kg': (DataLoaderKG, TextModelKG),
                                 'lgcn': (DataLoaderLightGCN, LightGCN),
                                 'lgcn_single': (DataLoaderLightGCN, LightGCNSingle),
                                 'lgcn_weights': (DataLoaderLightGCN, LightGCNWeight),
                                 'lgcn_single_attn': (DataLoaderLightGCN, LightSingleGCNAttn),
                                 'lgcn_attn': (DataLoaderLightGCN, LightGCNAttn),
                                 'reviews': (DataLoaderReviews, TextModelReviews),
                                 'ngcf': (DataLoaderReviews, NGCF),
                                 }[args.model]

    parallel_main()
    # main()
