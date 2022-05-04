import numpy as np
import torch
import torch.optim as opt
from torch.utils.data import DataLoader

from base_model import Single, BaseModel
from dataset import BaseDataset
from non_text_models import TorchGeometric
from parser import parse_args
from kg_models import DatasetKG, TextModelKG
from reviews_models import DatasetReviews, TextModelReviews


def get_class(name):
    ''' create a correct class based on the name of the model '''

    Dataset = BaseDataset
    if name == 'lgcn':
        Model = BaseModel
    elif name == 'reviews':
        Dataset = DatasetReviews
        Model = TextModelReviews
    elif name == 'kg':
        Dataset = DatasetKG
        Model = TextModelKG
    else:
        Model = TorchGeometric
    classes = [Model]
    if args.single:
        classes.insert(0, Single)

    class Model(*classes):
        pass

    args.logger.info(f'Class: {classes}')
    return Dataset, Model


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Dataset, Model = get_class(args.model)

    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(args, dataset)

    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   verbose=(not args.quiet),
                                                   patience=5,
                                                   min_lr=1e-8)

    model(loader, optimizer, scheduler)

    if args.predict:
        model.predict(save=True)
