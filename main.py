from torch.utils.data import DataLoader
from transformers import set_seed

from TextGCN import (
    AdvSamplDataset,
    AdvSamplModel,
    BaseDataset,
    LightGCN,
    LTRDataset,
    LTRLinear,
    LTRLinearWPop,
)
from TextGCN.parser import parse_args


def get_class(name):
    return {
        'LightGCN': [BaseDataset, LightGCN],
        'LTRLinear': [LTRDataset, LTRLinear],
        'LTRLinearWPop': [LTRDataset, LTRLinearWPop],
        'adv_sampling': [AdvSamplDataset, AdvSamplModel],
    }[name]


def main():

    args = parse_args()
    set_seed(args.seed)

    Dataset, Model = get_class(args.model)
    args.logger.info(f'Class: {Model}')
    args.logger.info(args)

    dataset = Dataset(args)
    model = Model(args, dataset)
    model.logger.info(model)

    if args.load:
        model.load(args.load)
        model.logger.info('Performance of the loaded model:')
        results = model.evaluate()
        model.print_metrics(results)

    if not args.no_train:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model.fit(loader)

    if args.predict:
        model.predict(range(dataset.n_users), with_scores=True, save=True)


if __name__ == '__main__':
    main()
