from rich import pretty, traceback

from dataloader import DataLoader
from lightgcn import LightGCN
from text_model import DataLoaderText, TextModel
from tripartite import DataLoaderTripartite, Tripartite

# from dataloader import DataLoaderText, DataLoaderLightGCN
# from lightgcn import LightGCN
# from text_model import TextModel

from parser import parse_args
from utils import set_seed

traceback.install()
pretty.install()


if __name__ == '__main__':

    args = parse_args()

    set_seed(args.seed)

    loader_class, model_class = {'text': (DataLoaderText, TextModel),
                                 'lightgcn': (DataLoader, LightGCN),
                                 'tripartite': (DataLoaderTripartite, Tripartite)
                                 }[args.model]

    dataset = loader_class(args)
    model = model_class(args, dataset)

    model.workout()
    if args.predict:
        predictions = model.predict(save=True)
