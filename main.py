from torch.utils.data import DataLoader

from .TextGCN import (
    AdvSamplDataset,
    AdvSamplModel,
    BaseDataset,
    BaseModel,
    LTRDataset,
    LTRGradientBoosted,
    LTRGradientBoostedWPop,
    LTRLinear,
    LTRLinearWPop,
    MarcusGradientBoosted,
    OneBatchDataset,
    TextData,
    TextModel
)
from .TextGCN.parser import parse_args
from .TextGCN.utils import seed_everything


def get_class(name):
    return {
        'lgcn': [BaseDataset, BaseModel],
        'adv_sampling': [AdvSamplDataset, AdvSamplModel],
        'text': [TextData, TextModel],
        'ltr_linear': [LTRDataset, LTRLinear],
        'ltr_pop': [LTRDataset, LTRLinearWPop],
        'xgboost': [OneBatchDataset, LTRGradientBoosted],
        'gbdt': [OneBatchDataset, LTRGradientBoosted],
        'gbdt_class': [OneBatchDataset, LTRGradientBoosted],
        'gbdt_pop': [OneBatchDataset, LTRGradientBoostedWPop],
        'xgboost_pop': [OneBatchDataset, LTRGradientBoostedWPop],
        'marcus': [OneBatchDataset, MarcusGradientBoosted]
    }[name]


if __name__ == '__main__':

    args = parse_args()
    seed_everything(args.seed)

    Dataset, Model = get_class(args.model)
    args.logger.info(f'Class: {Model}')
    args.logger.info(args)

    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(args, dataset)
    model.logger.info(model)

    model.fit(loader)
    if args.predict:
        model.predict(range(dataset.n_users), with_scores=True, save=True)
