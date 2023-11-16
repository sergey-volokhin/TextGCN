from torch.utils.data import DataLoader
from transformers import set_seed

from TextGCN import (
    AdvSamplDataset,
    AdvSamplModel,
    BaseDataset,
    BaseModel,
    LTRDataset,
    LTRLinear,
    LTRLinearWPop,
)
from TextGCN.parser import parse_args


def get_class(name):
    return {
        'lgcn': [BaseDataset, BaseModel],
        'adv_sampling': [AdvSamplDataset, AdvSamplModel],
        'ltr_linear': [LTRDataset, LTRLinear],
        'ltr_pop': [LTRDataset, LTRLinearWPop],
    }[name]


if __name__ == '__main__':

    args = parse_args()
    set_seed(args.seed)

    Dataset, Model = get_class(args.model)
    args.logger.info(f'Class: {Model}')
    args.logger.info(args)

    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(args, dataset)
    model.logger.info(model)

    if not args.no_train:
        model.fit(loader)

    if args.predict:
        model.predict(range(dataset.n_users), with_scores=True, save=True)
