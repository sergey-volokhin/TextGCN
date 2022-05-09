from torch_geometric import seed_everything
from torch.utils.data import DataLoader

from base_model import BaseModel
from dataset import BaseDataset
from non_text_models import TorchGeometric
from parser import parse_args
from kg_models import DatasetKG, TextModelKG
from reviews_models import DatasetReviews, TextModelReviews, TextData, TextModel
from LTR_reviews_models import LTRDataset, LTR


def get_class(name):
    return {
        'lgcn': [BaseDataset, BaseModel],
        'reviews': [DatasetReviews, TextModelReviews],
        'kg': [DatasetKG, TextModelKG],
        'ltr_kg': [LTRDataset, LTR],
        'ltr_reviews': [LTRDataset, LTR],
        'pos_u_neg_kg': [TextData, TextModel],
        'pos_u_neg_avg': [TextData, TextModel],
        'pos_avg_neg_kg': [TextData, TextModel],
        'pos_avg_neg_avg': [TextData, TextModel],
    }.get(name, [BaseDataset, TorchGeometric])


if __name__ == '__main__':

    args = parse_args()

    seed_everything(args.seed)

    Dataset, Model = get_class(args.model)
    args.logger.info(f'Class: {Model}')

    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Model(args, dataset)

    model.fit(loader)

    if args.predict:
        model.predict(save=True)
