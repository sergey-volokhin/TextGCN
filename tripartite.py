import numpy as np
import pandas as pd
import torch

from base_model import BaseModel
from dataloader import DataLoader
from utils import embed_text


class DataLoaderTripartite(DataLoader):

    def __init__(self, args):
        self._copy_args(args)
        self.bert_model = args.bert_model
        self._load_files()
        self._print_info()
        self._build_dicts()
        self._init_embeddings()
        # self._construct_graph()
        self._get_user_representation()

    def _copy_args(self, args):
        super()._copy_args(args)
        self.emb_batch_size = args.emb_batch_size

    def _load_files(self):
        super()._load_files()
        self.user_mapping = pd.read_csv(self.path + 'user_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.reviews = pd.read_table(self.path + 'reviews_text.tsv', header=0, names=['user_id', 'review', 'asin'])

    def _init_embeddings(self):
        ''' init user and item embeddings with reviews '''

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
        self.embedding_user = torch.nn.Embedding.from_pretrained(torch.tensor(user_vectors))

        item_vectors = {}
        item_mapping_dict = {i['org_id']: i['remap_id'] for i in self.item_mapping.to_dict(orient='records')}
        for item, group in self.reviews.groupby('asin')['vector']:
            item_vectors[item_mapping_dict[item]] = self.aggregation(group)
        item_vectors = np.array([i[1] for i in sorted(item_vectors.items())])
        self.embedding_item = torch.nn.Embedding.from_pretrained(torch.tensor(item_vectors))

    def aggregation(self, group):
        return group.mean(axis=0)


class Tripartite(BaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        self.load_model()

        self.to(self.device)
        self._build_optimizer()
        self.logger.info(self)

    def _copy_args(self, args):
        super()._copy_args(args)

    def get_representation(self):
        ''' calculating the node representation using all the layers'''
        pass
