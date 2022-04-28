import numpy as np
import torch
import torch.optim as opt
from torch.utils.data import DataLoader

from geom_models import TorchGeometric
from dataset import BaseDataset
from lightgcn import LightGCN
from parser import parse_args
from text_model_desc import DatasetKG, TextModelKG
from text_model_reviews import DatasetReviews, ReviewModel
from text_model_reviews_loss import TextModelReviewsLoss


def get_class(name, logger):
    ''' create a correct class based on the name of the model '''

    Dataset = BaseDataset
    if name == 'lgcn':
        Model = LightGCN
    elif name == 'reviews':
        Dataset = DatasetReviews
        Model = ReviewModel
    elif name == 'reviews_loss':
        Dataset = DatasetReviews
        Model = TextModelReviewsLoss
    elif name == 'kg':
        Dataset = DatasetKG
        Model = TextModelKG
    elif name in ['gcn', 'graphsage', 'gat', 'gatv2']:
        Model = TorchGeometric
    else:
        raise AttributeError('incorrect model')

    logger.info(f'Class: {Model}')

    return Dataset, Model


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Dataset, Model = get_class(args.model, args.logger)

    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(args, dataset)
    model = model.to(args.device)

    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   verbose=(not args.quiet),
                                                   patience=5,
                                                   min_lr=1e-8)

    model(loader, optimizer, scheduler)

    if args.predict:
        model.module.predict(save=True)
