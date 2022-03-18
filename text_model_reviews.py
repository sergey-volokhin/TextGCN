import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from base_model import BaseModel
from dataloader import DataLoader
from utils import embed_text


class DataLoaderReviews(DataLoader):

    def _load_files(self):
        super()._load_files()

        self.user_mapping = pd.read_csv(self.path + 'user_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.item_mapping = pd.read_csv(self.path + 'item_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.reviews = pd.read_table(self.path + 'reviews_text.tsv', dtype=str)[['asin', 'user_id', 'review']]

        ''' number of reviews to use for representing items and users'''
        self.num_reviews = int(pd.concat([self.reviews.groupby('user_id')['asin'].agg(len),
                                          self.reviews.groupby('asin')['user_id'].agg(len)]).median())

        ''' remove extra reviews not used in representation '''
        top_median_reviews = set()
        for _, group in self.reviews.groupby('user_id')['review']:
            top_median_reviews |= set(group.head(self.num_reviews))
        for _, group in self.reviews.groupby('asin')['review']:
            top_median_reviews |= set(group.head(self.num_reviews))
        self.reviews = self.reviews[self.reviews['review'].isin(top_median_reviews)]


class TextModelReviews(BaseModel):

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

    def _build_model(self):
        ''' pooling layers to combine lgcn and bert '''

        # linear combination of reviews' vectors
        self.review_agg_layer = torch.nn.Linear(in_features=self.num_reviews, out_features=1, bias=True)
        torch.nn.init.xavier_normal_(self.review_agg_layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(self.review_agg_layer.bias, 0.0)

        # linear combination of lightgcn and bert vectors
        self.linear = torch.nn.Linear(in_features=2, out_features=1, bias=True)
        torch.nn.init.xavier_normal_(self.linear.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(self.linear.bias, 0.0)

    def layer_aggregation(self, vectors):
        ''' aggregate layer representations into one vector '''
        return torch.mean(torch.stack(vectors, dim=1), dim=1)  # TODO could improve. attention/learn weights?

    def pad_reviews(self, reviews):
        ''' pad or cut existing reviews to have exactly self.num_reviews '''
        reviews = torch.tensor(np.stack(reviews)[:self.num_reviews]).float()
        return F.pad(reviews, (0, 0, 0, self.num_reviews - reviews.shape[0]))

    def get_representation(self):

        ''' recalculate embeddings '''  # TODO use (users, pos, neg) to save time
        if self.phase == 1:
            h = torch.cat([self.embedding_user.weight, self.embedding_item.weight])

        elif self.phase == 2:
            # pass reviews' embeddings through linear combination layer
            # then combine with LightGCN vector

            user_vectors = self.review_agg_layer(self.user_text_vectors.transpose(-1, 1)).squeeze()
            u_stacked = torch.stack((self.embedding_user.weight, user_vectors), axis=1)
            emb_user = self.linear(u_stacked.transpose(-1, 1)).squeeze()

            item_vectors = self.review_agg_layer(self.item_text_vectors.transpose(-1, 1)).squeeze()
            i_stacked = torch.stack((self.embedding_item.weight, item_vectors), axis=1)
            emb_item = self.linear(i_stacked.transpose(-1, 1)).squeeze()

            h = torch.cat([emb_user, emb_item])

        ''' pull the embeddings through all the layers and aggregate '''
        norm_matrix = self._dropout_norm_matrix()
        node_embed_cache = [h]
        for _ in range(self.n_layers):
            h = torch.sparse.mm(norm_matrix, h)
            node_embed_cache.append(h)
        node_embed_cache = self.layer_aggregation(node_embed_cache)
        return torch.split(node_embed_cache, [self.n_users, self.n_items])

    def setup_second_phase(self):
        '''
            this function is in the model and not in dataloader
            because we want to run the reliable lightgcn part first
        '''

        ''' calculate reviews' embeddings '''
        self.reviews['vector'] = embed_text(self.reviews['review'],
                                            'reviews',
                                            self.path,
                                            self.bert_model,
                                            self.emb_batch_size,
                                            self.device,
                                            self.logger)

        ''' make a table of embedded reviews for user and item to be pooled '''
        user_text_vectors = {}
        user_mapping_dict = {i['org_id']: i['remap_id'] for i in self.user_mapping.to_dict(orient='records')}
        for user, group in self.reviews.groupby('user_id')['vector']:
            user_text_vectors[user_mapping_dict[user]] = self.pad_reviews(group.values)
        self.user_text_vectors = torch.stack([i[1] for i in sorted(user_text_vectors.items())]).to(self.device)

        item_text_vectors = {}
        item_mapping_dict = {i['org_id']: i['remap_id'] for i in self.item_mapping.to_dict(orient='records')}
        for item, group in self.reviews.groupby('asin')['vector']:
            item_text_vectors[item_mapping_dict[item]] = self.pad_reviews(group.values)
        self.item_text_vectors = torch.stack([i[1] for i in sorted(item_text_vectors.items())]).to(self.device)

    def workout(self):

        if self.current_epoch <= self.total_epochs // 2:
            ''' first phase: train lightgcn embs for half epochs '''
            self.logger.info('Phase 1: Training LightGCN weights')
            self.phase = 1
            self.epochs //= 2
            super().workout()

        if self.current_epoch <= self.total_epochs:
            ''' second phase: pool trained with textual embeddings for half epochs '''
            self.logger.info('Phase 2: Training with pooled textual weights')
            self.phase = 2
            self.epochs = self.total_epochs
            self.current_epoch += 1
            self.setup_second_phase()
            super().workout()
