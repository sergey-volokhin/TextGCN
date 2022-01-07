import os

from dataloader import DataLoader
from model import Model
from parser import parse_args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dataset = DataLoader(args, seed=args.seed)
    model = Model(args, dataset)
    model.workout()
    if args.predict:
        model.predict()
