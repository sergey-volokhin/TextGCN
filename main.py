from torch.utils.data import DataLoader

from advanced_sampling import AdvSamplDataset, AdvSamplModel
from base_model import BaseModel
from dataset import BaseDataset
from kg_models import DatasetKG, TextModelKG
from ltr_models import LTRDataset, LTRLinear, LTRLinearWPop, LTRGBDT
from reviews_models import DatasetReviews, TextModelReviews
from text_joint_model import TextData, TextModel

from parser import parse_args
from utils import seed_everything


def get_class(name):
    return {
        'lgcn': [BaseDataset, BaseModel],
        'reviews': [DatasetReviews, TextModelReviews],
        'kg': [DatasetKG, TextModelKG],
        'text': [TextData, TextModel],
        'ltr_linear': [LTRDataset, LTRLinear],
        'ltr_pop': [LTRDataset, LTRLinearWPop],
        'adv_sampling': [AdvSamplDataset, AdvSamplModel],
        'gbdt': [LTRDataset, LTRGBDT],
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
