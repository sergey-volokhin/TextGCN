import pandas as pd
import torch
from tqdm import tqdm

from dataset import BaseDataset
from text_base_model import TextBaseModel
from kg_models import DatasetKG
from utils import embed_text


class DatasetReviews(BaseDataset):

    def __init__(self, args):
        super().__init__(args)
        self._load_reviews()
        self._calc_review_embs(args.emb_batch_size, args.bert_model)
        self._get_items_as_avg_reviews()

    def _load_reviews(self):
        self.reviews = pd.read_table(self.path + 'reviews_text.tsv', dtype=str)
        self.reviews = self.reviews[['asin', 'user_id', 'review', 'time']].sort_values(['asin', 'user_id'])
        self.reviews['asin'] = self.reviews['asin'].map(dict(self.item_mapping.values)).astype(int)
        self.reviews['user_id'] = self.reviews['user_id'].map(dict(self.user_mapping.values)).astype(int)

    def _calc_review_embs(self, emb_batch_size, bert_model):
        ''' load/calc embeddings of the reviews and setup the dicts '''
        emb_file = f'{self.path}/embeddings/item_full_reviews_loss_repr_{bert_model.split("/")[-1]}.torch'
        self.reviews['vector'] = embed_text(self.reviews['review'],
                                            emb_file,
                                            bert_model,
                                            emb_batch_size,
                                            self.device,
                                            self.logger).cpu().numpy().tolist()
        self.reviews_vector = self.reviews.set_index(['asin', 'user_id'])['vector']

    def _get_items_as_avg_reviews(self):
        ''' use average of reviews to represent items '''

        # number of reviews to use for representing items and users
        group_user = self.reviews.groupby('user_id')
        group_item = self.reviews.groupby('asin')
        self.num_reviews = int(pd.concat([group_item['user_id'].agg(len),
                                          group_user['asin'].agg(len)]).median())

        # use only most recent reviews for representation
        cut_reviews = set()
        for _, group in tqdm(group_user,
                             leave=False,
                             desc='selecting reviews',
                             dynamic_ncols=True,
                             disable=self.slurm):
            cut_reviews |= set(group.sort_values('time', ascending=False)['review'].head(self.num_reviews))
        for _, group in tqdm(group_item,
                             leave=False,
                             desc='selecting reviews',
                             dynamic_ncols=True,
                             disable=self.slurm):
            cut_reviews |= set(group.sort_values('time', ascending=False)['review'].head(self.num_reviews))

        item_text_embs = {}
        self.top_med_reviews = self.reviews[self.reviews['review'].isin(cut_reviews)]
        for item, group in self.top_med_reviews.groupby('asin')['vector']:
            item_text_embs[item] = torch.tensor(group.values.tolist()).mean(axis=0)
        self.items_as_avg_reviews = torch.stack(self.item_mapping['remap_id'].map(
            item_text_embs).values.tolist()).to(self.device)


class TextModelReviews(TextBaseModel):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.reviews_vector = dataset.reviews_vector
        self.items_as_avg_reviews = dataset.items_as_avg_reviews

    def bert_sim(self, users, pos, neg):

        # # represent pos items with respective user's review
        # cands = torch.tensor(self.reviews_vector.loc[torch.stack([pos, users], axis=1).tolist()].values.tolist())

        # represent pos items with mean of their reviews
        cands = self.items_as_avg_reviews[pos.cpu()]

        # represent neg items with mean of their reviews
        refs = self.items_as_avg_reviews[neg.cpu()]

        return self.sim_fn(cands, refs).to(self.device)


class TextData(DatasetKG, DatasetReviews):
    pass


class TextModel(TextModelReviews):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.items_as_desc = dataset.items_as_desc

    def bert_sim(self, users, pos, neg):
        cands = torch.tensor(self.reviews_vector.loc[torch.stack([pos, users], axis=1).tolist()].values.tolist()).to(
            self.device)
        refs = self.items_as_desc[neg.cpu()]
        return self.sim_fn(cands, refs).to(self.device)
