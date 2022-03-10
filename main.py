import cProfile
import pstats

import numpy as np
import torch
from rich import pretty, traceback

from dataloader import DataLoader as DataLoaderLightGCN
from lightgcn import LightGCN
from parser import parse_args
from text_model import DataLoaderText, TextModel
from text_model_reviews import DataLoaderReviews, TextModelReviews

# traceback.install()
pretty.install()


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loader_class, model_class = {'text': (DataLoaderText, TextModel),
                                 'lightgcn': (DataLoaderLightGCN, LightGCN),
                                 'reviews': (DataLoaderReviews, TextModelReviews)
                                 }[args.model]

    # profiler = cProfile.Profile()
    # profiler.enable()

    dataset = loader_class(args)
    model = model_class(args, dataset)

    model.workout()
    if args.predict:
        predictions = model.predict(save=True)

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
