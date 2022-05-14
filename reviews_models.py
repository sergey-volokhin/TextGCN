import pandas as pd
import torch
from tqdm.auto import tqdm

from dataset import BaseDataset
from text_base_model import TextBaseModel
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
                                            self.device).cpu().numpy().tolist()
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
        # saving top_med_reviews to model so we could extend LTR
        self.top_med_reviews = self.reviews[self.reviews['review'].isin(cut_reviews)]
        for item, group in self.top_med_reviews.groupby('asin')['vector']:
            item_text_embs[item] = torch.tensor(group.values.tolist()).mean(axis=0)
        items_as_avg_reviews = self.item_mapping['remap_id'].map(item_text_embs).values.tolist()
        self.items_as_avg_reviews = torch.stack(items_as_avg_reviews).to(self.device)


class TextModelReviews(TextBaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        ''' how do we represent items in sampled triplets '''
        if args.pos == 'avg' or args.model == 'reviews':
            self.pos_item_reprs = self.get_item_reviews_mean
        elif args.pos == 'user':
            self.pos_item_reprs = self.get_item_reviews_user

        if args.neg == 'avg' or args.model == 'reviews':
            self.neg_item_reprs = self.get_item_reviews_mean

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.reviews_vector = dataset.reviews_vector
        self.items_as_avg_reviews = dataset.items_as_avg_reviews

    def get_item_reviews_mean(self, users, items):
        ''' represent items with mean of their reviews '''
        return self.items_as_avg_reviews[items]

    def get_item_reviews_user(self, users, items):
        ''' represent items with the review of corresponding user '''
        df = self.reviews_vector.loc[torch.stack([items, users], axis=1).tolist()]
        return torch.tensor(df.values.tolist()).to(self.device)
