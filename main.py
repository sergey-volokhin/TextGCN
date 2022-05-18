from torch.utils.data import DataLoader

from base_model import BaseModel
from dataset import BaseDataset
from kg_models import DatasetKG, TextModelKG
from ltr_models import LTRCosine, LTRDataset, LTRLinear, LTRLinearFeatures, LTRSimple
from non_text_models import TorchGeometric
from parser import parse_args
from reviews_models import DatasetReviews, TextModelReviews
from text_joint_model import TextData, TextModel
from utils import seed_everything


def get_class(name):
    return {
        'lgcn': [BaseDataset, BaseModel],
        'reviews': [DatasetReviews, TextModelReviews],
        'kg': [DatasetKG, TextModelKG],
        'text': [TextData, TextModel],
        'ltr_kg': [LTRDataset, LTRCosine],
        'ltr_reviews': [LTRDataset, LTRCosine],
        'ltr_simple': [LTRDataset, LTRSimple],
        'ltr_linear': [LTRDataset, LTRLinear],
        'ltr_linear_features': [LTRDataset, LTRLinearFeatures],
    }.get(name, [BaseDataset, TorchGeometric])


if __name__ == '__main__':

    args = parse_args()

    seed_everything(args.seed)

    Dataset, Model = get_class(args.model)
    args.logger.info(f'Class: {Model}')
    args.logger.info(args)
    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(args, dataset)

    model.fit(loader)

    if args.evaluate:
        model.evaluate(-1)

    if args.predict:
        model.predict(save=True)
