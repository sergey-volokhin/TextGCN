from dataloader import DataLoaderText, DataLoaderLightGCN
from text_model import TextModel
from lightgcn import LightGCN
from parser import parse_args
from utils import set_seed


if __name__ == '__main__':

    args = parse_args()

    set_seed(args.seed)

    loader_class = {'text': DataLoaderText, 'lightgcn': DataLoaderLightGCN}[args.model]
    model_class = {'text': TextModel, 'lightgcn': LightGCN}[args.model]

    dataset = loader_class(args)
    model = model_class(args, dataset)

    model.workout()
    if args.predict:
        predictions = model.predict(save=True)
