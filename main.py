import numpy as np
import torch
import torch.optim as opt
from torch import nn
from torch.utils.data import DataLoader

from dataset import BaseDataset
from lightgcn import (LightGCN, LightGCNAttn, LightGCNSingle, LightGCNWeight,
                      LightGCNSingleAttn)
# from text_model import DatasetKG, TextModelKG
from text_model_reviews import NGCF, DatasetReviews, ReviewModel, ReviewModelSingle
from parser import parse_args


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Dataset, Model = {
        'lgcn': (BaseDataset, LightGCN),
        'lgcn_single': (BaseDataset, LightGCNSingle),
        'lgcn_attn': (BaseDataset, LightGCNAttn),
        'lgcn_single_attn': (BaseDataset, LightGCNSingleAttn),
        'lgcn_weights': (BaseDataset, LightGCNWeight),
        'reviews': (DatasetReviews, ReviewModel),
        'reviews_single': (DatasetReviews, ReviewModelSingle),
        'ngcf': (DatasetReviews, NGCF),
        # 'kg': (DatasetKG, TextModelKG),
    }[args.model]

    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(args, dataset)

    model = nn.DataParallel(model, device_ids=range(len(args.gpu.split(','))))
    model = model.to(torch.device("cuda"))

    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   verbose=(not args.quiet),
                                                   patience=5,
                                                   min_lr=1e-6)

    model(loader, optimizer, scheduler)

    if args.predict:
        model.module.predict(save=True)
