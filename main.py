import numpy as np
import torch
import torch.optim as opt
from torch import nn
from torch.utils.data import DataLoader

from base_model import NGCF, Single
from dataset import BaseDataset
from lightgcn import LightGCN
from graphsage import GraphSageMean, GraphSagePool
from parser import parse_args
from text_model_desc import DatasetKG, TextModelKG
from text_model_reviews import DatasetReviews, ReviewModel
from text_model_reviews_loss import TextModelReviewsLoss


def get_class(name, logger):
    ''' create a correct class based on the name of the model '''

    if name == 'lgcn':
        Dataset = BaseDataset
        classes = [LightGCN]
    elif name == 'graphsage_mean':
        Dataset = BaseDataset
        classes = [GraphSageMean]
    elif name == 'graphsage_pool':
        Dataset = BaseDataset
        classes = [GraphSagePool]
    elif name == 'reviews':
        Dataset = DatasetReviews
        classes = [ReviewModel]
    elif name == 'reviews_loss':
        Dataset = DatasetReviews
        classes = [TextModelReviewsLoss]
    elif name == 'kg':
        Dataset = DatasetKG
        classes = [TextModelKG]
    else:
        raise AttributeError('incorrect model')

    if args.single:
        classes.insert(0, Single)
    if args.ngcf:
        classes.insert(0, NGCF)
    logger.info(f'Classes: {classes}')

    class Model(*classes):
        pass

    return Dataset, Model


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Dataset, Model = get_class(args.model, args.logger)

    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(args, dataset)
    if len(args.gpu.split(',')) > 1:
        model = nn.DataParallel(model, device_ids=range(len(args.gpu.split(','))))
    model = model.to(args.device)

    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   verbose=(not args.quiet),
                                                   patience=5,
                                                   min_lr=1e-8)

    model(loader, optimizer, scheduler)

    if args.predict:
        model.module.predict(save=True)
