from torch.utils.data import DataLoader
from transformers import set_seed

from src.advanced_sampling import AdvSamplDataset, AdvSamplModel
from src.DatasetRanking import DatasetRanking
from src.DatasetScoring import DatasetScoring
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
        'LightGCNRank': [DatasetRanking, LightGCNRank],
        'LightGCNScore': [DatasetScoring, LightGCNScore],
        'LTRLinearRank': [LTRDatasetRank, LTRLinearRank],
        'LTRLinearRankWPop': [LTRDatasetRank, LTRLinearRankWPop],
        'LTRLinearScore': [LTRDatasetScore, LTRLinearScore],
        'LTRLinearScoreWPop': [LTRDatasetScore, LTRLinearScoreWPop],
        'adv_sampling': [AdvSamplDataset, AdvSamplModel],
    }[name]


def main():

    config = parse_args()
    set_seed(config.seed)
    Dataset, Model = get_class(config.model)
    config.logger.info(f'Class: {Model.__name__}')
    config.logger.info('Parameters:')
    for key, value in vars(config).items():
        config.logger.info(f'  {key}: {value}')
    config.logger.info('')

    dataset = Dataset(config)
    model = Model(config, dataset)
    model.logger.info(model)

    if config.load:
        model.load(config.load)
        model.logger.info('Performance of the loaded model:')
        results = model.evaluate()
        model.metrics_log.log(results)

    if not config.no_train:
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        model.fit(loader)

    if config.predict:
        if 'Rank' in config.model:
            model.predict(users=range(dataset.n_users), save=True, with_scores=True)
        elif 'Score' in config.model:
            model.predict(model.test_df, save=True)


if __name__ == '__main__':
    main()
