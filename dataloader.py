import os
import random

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import get_logger, tokenize_text, embed_text


class DataLoader(object):

    def __init__(self, args, seed=False):

        self.logger = get_logger(args)
        self.device = args.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)

        self.args = args
        self.path = args.datapath
        self._load_files()
        self._get_numbers()
        self._print_info()
        self._construct_text_representation()
        self._construct_embeddings()
        self._construct_graphs()

        self.batch_size = min(args.batch_size, self.n_train)
        self.num_batches = (self.n_train - 1) // self.batch_size + 1
        self.embed_batch_size = args.batch_size if torch.cuda.is_available() and args.gpu else 16

    def _load_files(self):
        self.logger.info('loading data')
        self.train_df = pd.read_table(f'{self.path}/train.tsv', dtype=np.int32, header=0, names=['user_id', 'asin']).sort_values(by=['user_id'])
        self.test_df = pd.read_table(f'{self.path}/test.tsv', dtype=np.int32, header=0, names=['user_id', 'asin']).sort_values(by=['user_id'])
        self.kg_df_text = pd.read_table(f'{self.path}/kg_readable.tsv', dtype=str)[['asin', 'relation', 'attribute']]
        self.user_mapping = pd.read_csv(f'{self.path}/user_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.item_mapping = pd.read_csv(f'{self.path}/item_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.user_mapping['remap_id'] += max(self.item_mapping['remap_id']) + 1
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].apply(np.array).to_dict()
        self.test_user_dict = self.test_df.groupby('user_id')['asin'].apply(np.array).to_dict()

    def _get_numbers(self):
        self.items = set(self.train_df['asin'].unique()) | set(self.test_df['asin'].unique())
        self.users = set(self.train_df['user_id'].unique()) | set(self.test_df['user_id'].unique())
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.users = list(self.users)
        assert len(self.kg_df_text['asin'].unique()) == self.n_items, 'number of items in KG not equal to number of items in train'

    def _print_info(self):
        self.logger.info(f'n_users:      {self.n_users:-7}')
        self.logger.info(f'n_items:      {self.n_items:-7}')
        self.logger.info(f'n_entities:   {self.n_users + self.n_items:-7}')
        self.logger.info(f'n_train:      {self.n_train:-7}')
        self.logger.info(f'n_test:       {self.n_test:-7}')

    def _construct_text_representation(self):
        self.item_text_dict = {}
        for asin, group in tqdm(self.kg_df_text.groupby('asin'), desc='construct text repr', dynamic_ncols=True):
            vals = group[['relation', 'attribute']].values
            self.item_text_dict[asin] = f' {self.args.sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping['text'] = self.item_mapping['org_id'].map(self.item_text_dict)

    def _construct_embeddings(self):

        ''' construct text representations for items and embed them with BERT '''
        embeddings = embed_text(self.item_mapping['text'].to_list(), self.path, self.args.bert_model, self.embed_batch_size)

        ''' randomly initialize all entity embeddings, we will overwrite the item embeddings next '''
        self.entity_embeddings = nn.Embedding(self.n_items + self.n_users, self.args.embed_size).to(self.device)
        if self.args.single_vector:
            self.user_vector = nn.parameter.Parameter(torch.Tensor(self.args.embed_size))
            nn.init.xavier_uniform_(self.user_vector.unsqueeze(1))
        else:
            self.user_vector = None

        with torch.no_grad():
            self.entity_embeddings.weight[self.item_mapping['remap_id']] = embeddings

    def _construct_graphs(self):
        ''' create bipartite graph with initial vectors '''
        self.graph = dgl.heterograph({('user', 'bought', 'item'): (self.train_df['user_id'].values, self.train_df['asin'].values),
                                      ('item', 'bought_by', 'user'): (self.train_df['asin'].values, self.train_df['user_id'].values)}).to(self.device)
        user_ids = torch.LongTensor(self.user_mapping['remap_id']).to(self.device)
        item_ids = torch.LongTensor(self.item_mapping['remap_id']).to(self.device)
        self.graph.ndata['e'] = {'user': self.entity_embeddings(user_ids),
                                 'item': self.entity_embeddings(item_ids)}
        self.graph.ndata['id'] = {'user': user_ids,
                                  'item': item_ids}

    def sampler(self):
        for _ in range(self.num_batches):
            if self.batch_size <= self.n_users:
                users = random.sample(self.users, self.batch_size)
            else:
                users = random.choices(self.users, k=self.batch_size)
            pos_items, neg_items = [], []
            for u in users:
                pos_items.append(random.choice(self.train_user_dict[u]))
                while True:
                    neg_i_id = random.randrange(self.n_items)
                    if neg_i_id not in self.train_user_dict[u]:
                        break
                neg_items.append(neg_i_id)
            yield users, pos_items, neg_items
