import cProfile
import pstats

import numpy as np
import torch
from rich import pretty

from torch.utils.data import DataLoader
from dataloader import MyDataLoader as DataLoaderLightGCN
from lightgcn import LightGCN, LightGCNAttn, LightGCNWeight, LightGCNSingle
from parser import parse_args
from text_model import DataLoaderKG, TextModelKG
from text_model_reviews import DataLoaderReviews, TextModelReviews, NGCF

pretty.install()


def main():
    dataset = loader_class(args)
    model = model_class(args, dataset)
    model()
    if args.predict:
        model.predict(save=True)


def parallel_main():
    # dataset = loader_class(args)
    # DataLoader(dataset)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=[0])
    pass


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
                                 'lgcn_weigths': (DataLoaderLightGCN, LightGCNWeight),
                                 'lightattn': (DataLoaderLightGCN, LightGCNAttn),
                                 'reviews': (DataLoaderReviews, TextModelReviews),
                                 'ngcf': (DataLoaderReviews, NGCF),
                                 }[args.model]

    main()
