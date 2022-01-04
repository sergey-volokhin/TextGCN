import collections
import os
import random

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import embed_text, init_bert


class DataLoader(object):
    def __init__(self, args, seed=1234):
        random.seed(seed)
        torch.manual_seed(seed)
        self.args = args
        self.path = args.datapath
        self.device = args.device
        self.logger = args.logger
        self._load_files()
        self._get_numbers()
        self._print_info()
        self._construct_emb()
        self._construct_graph()

    def _load_files(self):
        self.logger.info('loading data')
        self.train_df = pd.read_table(f'{self.path}/train.tsv', dtype=np.int32).sort_values(by=['user_id'])
        self.test_df = pd.read_table(f'{self.path}/test.tsv', dtype=np.int32).sort_values(by=['user_id'])
        self.kg_df_text = pd.read_table(f'{self.path}/kg_readable.tsv')
        self.user_mapping = pd.read_csv(f'{self.path}/../user_list.txt', sep=' ')
        self.item_mapping = pd.read_csv(f'{self.path}/../item_list.txt', sep=' ')
        self.train_user_dict = self.train_df.groupby('user_id')['item_id'].apply(np.array).to_dict()
        self.test_user_dict = self.test_df.groupby('user_id')['item_id'].apply(np.array).to_dict()

    def _get_numbers(self):
        self.entities = set(self.kg_df_text['asin'].unique()) | \
            set(self.train_df['user_id'].unique()) | \
            set(self.train_df['item_id'].unique())
        self.n_users = len(set(self.train_df['user_id'].unique()) | set(self.test_df['user_id'].unique()))
        self.n_items = len(set(self.train_df['item_id'].unique()) | set(self.test_df['item_id'].unique()))
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.n_entities = len(self.entities)

    def _print_info(self):
        self.logger.info(f'n_users:      {self.n_users:-7}')
        self.logger.info(f'n_items:      {self.n_items:-7}')
        self.logger.info(f'n_entities:   {self.n_entities:-7}')
        self.logger.info(f'n_train:      {self.n_train:-7}')
        self.logger.info(f'n_test:       {self.n_test:-7}')

    def _construct_emb(self):
        ''' random init for users, overwrite for items '''
        self.entity_embeddings = nn.Embedding(self.n_entities, self.args.embed_size)

        ''' construct text representations for items and embed them with BERT '''
        self.item_text_dict = {}
        for asin, group in self.kg_df_text.groupby('asin'):
            vals = group[['relation', 'attribute']].values
            self.item_text_dict[asin] = ' [SEP] '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping['remap_id'] += max(self.user_mapping['remap_id']) + 1
        self.item_mapping['text'] = self.item_mapping.apply(lambda x: self.item_text_dict[x['org_id']], axis=1)
        with torch.no_grad():
            self.entity_embeddings.weight[self.item_mapping['remap_id']] = embed_text(self.item_mapping['text'].to_list(), *init_bert(self.args))

    def _construct_graph(self):
        ''' create bipartite graph with initial vectors '''
        self.graph = dgl.heterograph({('user', 'bought', 'item'): (self.train_df.values[:, 0], self.train_df.values[:, 1]),
                                      ('item', 'bought_by', 'user'): (self.train_df.values[:, 1], self.train_df.values[:, 0])})
        self.graph.ndata['e'] = {'item': self.entity_embeddings(torch.LongTensor(self.item_mapping['remap_id'])),
                                 'user': self.entity_embeddings(torch.LongTensor(self.user_mapping['remap_id']))}
        self.graph.ndata['id'] = {'user': torch.arange(min(self.user_mapping['remap_id']), max(self.user_mapping['remap_id']) + 1, dtype=torch.long).to(self.device),
                                  'item': torch.arange(min(self.item_mapping['remap_id']), max(self.item_mapping['remap_id']) + 1, dtype=torch.long).to(self.device)}

    # def create_edge_sampler(self, graph, **kwargs):
    #     edge_sampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
    #     return edge_sampler(graph,
    #                         neg_sample_size=1,
    #                         shuffle=True,
    #                         return_false_neg=True,
    #                         exclude_positive=False,
    #                         **kwargs)

    # def sampler(self, batch_size, pos_mode, num_workers=8):
    #     if batch_size < 0 or batch_size > self.n_train:
    #         batch_size = self.n_train
    #         n_batch = 1
    #     else:
    #         n_batch = self.n_train // batch_size + 1

    #     if pos_mode == 'unique':
    #         exist_users = list(self.train_user_dict)
    #         i = 0
    #         while i < n_batch:
    #             i += 1
    #             if batch_size <= self.n_users:
    #                 users = random.sample(exist_users, batch_size)
    #             else:
    #                 users = random.choices(exist_users, k=batch_size)
    #             pos_items, neg_items = [], []
    #             for u in users:
    #                 pos_items.append(random.choice(self.train_user_dict[u]))
    #                 while True:
    #                     neg_i_id = random.randrange(self.n_items)
    #                     if neg_i_id not in self.train_user_dict[u]:
    #                         break
    #                 neg_items.append(neg_i_id)
    #             yield users, pos_items, neg_items, None
    #     else:
    #         for pos_g, neg_g in self.create_edge_sampler(self.cf_graph,
    #                                                      batch_size=batch_size,
    #                                                      num_workers=num_workers,
    #                                                      negative_mode='head'):
    #             false_neg = neg_g.edata['false_neg']
    #             pos_g.copy_from_parent()
    #             neg_g.copy_from_parent()
    #             i_idx, u_idx = pos_g.all_edges(order='eid')
    #             neg_i_idx, _ = neg_g.all_edges(order='eid')
    #             users = torch.LongTensor(pos_g.ndata['id'][u_idx]).to(self.device)
    #             pos_items = torch.LongTensor(pos_g.ndata['id'][i_idx]).to(self.device)
    #             neg_items = torch.LongTensor(neg_g.ndata['id'][neg_i_idx]).to(self.device)
    #             yield users, pos_items, neg_items, false_neg
