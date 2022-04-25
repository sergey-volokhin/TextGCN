# borked

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from base_model import BaseModel
from dataset import BaseDataset
from utils import embed_text


class DatasetReviews(BaseDataset):

    def _load_files(self):
        super()._load_files()

        self.user_mapping = pd.read_csv(self.path + 'user_list.txt', sep=' ', dtype=str)[['org_id', 'remap_id']]
        self.item_mapping = pd.read_csv(self.path + 'item_list.txt', sep=' ', dtype=str)[['org_id', 'remap_id']]
        self.reviews = pd.read_table(self.path + 'reviews_text.tsv', dtype=str)[['asin', 'user_id', 'review', 'time']]

        ''' number of reviews to use for representing items and users '''
        self.num_reviews = int(pd.concat([self.reviews.groupby('user_id')['asin'].agg(len),
                                          self.reviews.groupby('asin')['user_id'].agg(len)]).median())

        ''' use only most recent reviews for representation '''
        cut_reviews = set()
        for _, group in tqdm(self.reviews.groupby('user_id'),
                             leave=False,
                             desc='selecting reviews',
                             dynamic_ncols=True,
                             disable=self.slurm):
            cut_reviews |= set(group.sort_values('time', ascending=False)['review'].head(self.num_reviews))
        for _, group in tqdm(self.reviews.groupby('asin'),
                             leave=False,
                             desc='selecting reviews',
                             dynamic_ncols=True,
                             disable=self.slurm):
            cut_reviews |= set(group.sort_values('time', ascending=False)['review'].head(self.num_reviews))
        self.reviews = self.reviews[self.reviews['review'].isin(cut_reviews)]


class ReviewModel(BaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.phase = 1
        self.epochs //= 2

    def _copy_args(self, args):
        super()._copy_args(args)
        self.path = args.data
        self.bert_model = args.bert_model
        self.emb_batch_size = args.emb_batch_size

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.reviews = dataset.reviews
        self.num_reviews = dataset.num_reviews
        self.user_mapping = dataset.user_mapping
        self.item_mapping = dataset.item_mapping

    def _add_vars(self):
        super()._add_vars()

        ''' pooling layers to combine lgcn and bert '''
        gain = nn.init.calculate_gain('relu')

        # linear combination of reviews' vectors
        self.review_agg_layer = nn.Linear(in_features=self.num_reviews, out_features=1, bias=True, device=self.device)
        nn.init.xavier_normal_(self.review_agg_layer.weight, gain=gain)
        nn.init.constant_(self.review_agg_layer.bias, 0.0)

        # linear combination of lightgcn and bert vectors
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True, device=self.device)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.constant_(self.linear.bias, 0.0)

    def _init_embeddings(self, emb_size):
        super()._init_embeddings(emb_size)

        ''' load/calc embeddings of the reviews and setup the dicts '''

        emb_file = f'{self.path}/embeddings/item_kg_repr_{self.bert_model.split("/")[-1]}.torch'
        self.reviews['vector'] = embed_text(self.reviews['review'],
                                            emb_file,
                                            self.bert_model,
                                            self.emb_batch_size,
                                            self.device,
                                            self.logger)

        ''' pool embedded reviews for user and item to be represented '''
        user_text_embs = {}
        for user, group in self.reviews.groupby('user_id')['vector']:
            # user_text_embs[user] = self._pad_reviews(group.values)
            user_text_embs[user] = group.values.mean(axis=0)
        self.user_text_embs = nn.Parameter(torch.stack(self.user_mapping['org_id'].map(user_text_embs).values.tolist()))

        item_text_embs = {}
        for item, group in self.reviews.groupby('asin')['vector']:
            # item_text_embs[item] = self._pad_reviews(group.values)
            item_text_embs[item] = group.values.mean(axis=0)
        self.item_text_embs = nn.Parameter(torch.stack(self.item_mapping['org_id'].map(item_text_embs).values.tolist()))

    def _pad_reviews(self, reviews):
        ''' pad or cut existing reviews to have equal number per user/item '''
        reviews = torch.tensor(np.stack(reviews)[:self.num_reviews]).float()
        return F.pad(reviews, (0, 0, 0, self.num_reviews - reviews.shape[0]))

    def layer_aggregation(self, norm_matrix, curent_lvl_emb_matrix):
        return torch.sparse.mm(norm_matrix, curent_lvl_emb_matrix)

    def layer_combination(self, vectors):
        return torch.mean(torch.stack(vectors, dim=1), dim=1)

    @property
    def embedding_matrix(self):
        ''' get the embedding matrix of 0th layer

            for phase 1 (lightgcn):
                take embs as is

            for phase 2 (reviews):
                get linear combination of top num_reviews reviews,
                combine with LGCN vector using another linear layer
        '''

        if self.phase == 1:
            return super().embedding_matrix

        user_vectors = self.review_agg_layer(self.user_text_embs.transpose(-1, 1)).squeeze()
        u_stacked = torch.stack((self.embedding_user.weight, user_vectors), axis=1)
        emb_user = self.linear(u_stacked.transpose(-1, 1)).squeeze()

        item_vectors = self.review_agg_layer(self.item_text_embs.transpose(-1, 1)).squeeze()
        i_stacked = torch.stack((self.embedding_item.weight, item_vectors), axis=1)
        emb_item = self.linear(i_stacked.transpose(-1, 1)).squeeze()

        return torch.cat([emb_user, emb_item])

    def forward(self, loader, optimizer, scheduler):
        super().forward(loader, optimizer, scheduler)
        self.phase = 2
        super().forward(loader, optimizer, scheduler)
