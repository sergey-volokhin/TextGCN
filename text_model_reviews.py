import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from base_model import BaseModel
from dataloader import DataLoader
from utils import embed_text


class DataLoaderReviews(DataLoader):

    def _copy_args(self, args):
        super()._copy_args(args)

        self.sep = args.sep
        self.freeze = args.freeze
        self.bert_model = args.bert_model
        self.emb_batch_size = args.emb_batch_size

    def _load_files(self):
        super()._load_files()
        self.reviews = pd.read_table(self.path + 'reviews_text.tsv', dtype=str)[['asin', 'review', 'user_id']]
        self.user_mapping = pd.read_csv(self.path + 'user_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.item_mapping = pd.read_csv(self.path + 'item_list.txt', sep=' ')[['org_id', 'remap_id']]

    def _init_embeddings(self):
        super()._init_embeddings()

        ''' additional user and item embeddings with reviews text '''

        self.reviews['vector'] = embed_text(self.reviews['review'],
                                            self.path,
                                            self.bert_model,
                                            self.emb_batch_size,
                                            self.device).apply(np.array)

        user_vectors = {}
        user_mapping_dict = {i['org_id']: i['remap_id'] for i in self.user_mapping.to_dict(orient='records')}
        for user, group in self.reviews.groupby('user_id')['vector']:
            user_vectors[user_mapping_dict[user]] = self.aggregation(group)
        user_vectors = np.array([i[1] for i in sorted(user_vectors.items())])
        self.embedding_user_text = torch.nn.Embedding.from_pretrained(torch.tensor(user_vectors))

        item_vectors = {}
        item_mapping_dict = {i['org_id']: i['remap_id'] for i in self.item_mapping.to_dict(orient='records')}
        for item, group in self.reviews.groupby('asin')['vector']:
            item_vectors[item_mapping_dict[item]] = self.aggregation(group)
        item_vectors = np.array([i[1] for i in sorted(item_vectors.items())])
        self.embedding_item_text = torch.nn.Embedding.from_pretrained(torch.tensor(item_vectors))

    def aggregation(self, reviews):
        '''
            function that aggregates the reviews embeddings to create
            user and item representations
        '''
        return reviews.mean(axis=0)


class TextModelReviews(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.layer_size = args.layer_size
        self.single_vector = args.single_vector

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.train_user_dict = dataset.train_user_dict
        self.embedding_user_text = dataset.embedding_user_text
        self.embedding_item_text = dataset.embedding_item_text

    def get_representation(self):

        if self.phase == 1:
            h = torch.cat([self.embedding_user.weight,
                           self.embedding_item.weight])
        else:
            h = torch.cat([embedding_user.weight,
                           embedding_user_text.weight,
                           embedding_item.weight,
                           embedding_item_text.weight], axis=1)

        norm_matrix = self._dropout_norm_matrix()
        node_embed_cache = [h]
        for _ in range(self.n_layers):
            h = torch.sparse.mm(norm_matrix, h)
            node_embed_cache.append(h)
        node_embed_cache = self.aggregate_layers(node_embed_cache)
        return torch.split(node_embed_cache, [self.n_users, self.n_items])

    def aggregate_layers(self, vectors):
        return torch.mean(torch.stack(vectors, dim=1), dim=1)

    def workout(self):

        ''' first phase: train lightgcn embs '''
        self.phase = 1
        super().workout()

        ''' second phase: concat trained with textual embeddings '''
        self.phase = 2
        super().workout()
