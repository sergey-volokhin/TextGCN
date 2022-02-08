from dataloader import DataLoader
from model import Model
from parser import parse_args
from utils import set_seed


if __name__ == '__main__':

    args = parse_args()

    set_seed(args.seed)

    dataset = DataLoader(args)
    model = Model(args, dataset)

    model.workout()
    if args.predict:
        predictions = model.predict(save=True)
