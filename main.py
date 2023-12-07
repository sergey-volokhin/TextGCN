from torch.utils.data import DataLoader
from transformers import set_seed

from src.advanced_sampling import AdvSamplDataset, AdvSamplModel
from src.BaseDataset import BaseDataset
from src.DatasetRatings import DatasetRatings
from src.LightGCN import LightGCNRank, LightGCNScore
from src.LTRBaseModel import LTRDatasetRank, LTRDatasetScore
from src.LTRLinear import (
    LTRLinearRank,
    LTRLinearRankWPop,
    LTRLinearScore,
    LTRLinearScoreWPop,
)
from src.parsing import parse_args


def get_class(name):
    return {
        'LightGCNRank': [BaseDataset, LightGCNRank],
        'LightGCNScore': [DatasetRatings, LightGCNScore],
        'LTRLinearRank': [LTRDatasetRank, LTRLinearRank],
        'LTRLinearRankWPop': [LTRDatasetRank, LTRLinearRankWPop],
        'LTRLinearScore': [LTRDatasetScore, LTRLinearScore],
        'LTRLinearScoreWPop': [LTRDatasetScore, LTRLinearScoreWPop],
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
        model.predict(users=range(dataset.n_users), save=True, with_scores=True)


if __name__ == '__main__':
    main()
