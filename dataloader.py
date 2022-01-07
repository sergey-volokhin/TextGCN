import random

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import embed_text, get_logger, init_bert


class DataLoader(object):

    def __init__(self, args, seed=False):

        self.logger = get_logger(args)
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)

        self.args = args
        self.path = args.datapath
        self._load_files()
        self._get_numbers()
        self._print_info()
        self._construct_embeddings()
        self._construct_graph()

        self.batch_size = min(args.batch_size, self.n_train)
        self.num_batches = (self.n_train - 1) // self.batch_size + 1

    def _load_files(self):
        self.logger.info('loading data')
        self.train_df = pd.read_table(f'{self.path}/train.tsv', dtype=np.int32, header=0, names=['user_id', 'asin']).sort_values(by=['user_id'])
        self.test_df = pd.read_table(f'{self.path}/test.tsv', dtype=np.int32, header=0, names=['user_id', 'asin']).sort_values(by=['user_id'])
        self.kg_df_text = pd.read_table(f'{self.path}/kg_readable.tsv')[['asin', 'relation', 'attribute']]
        self.user_mapping = pd.read_csv(f'{self.path}/user_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.item_mapping = pd.read_csv(f'{self.path}/item_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.item_mapping['remap_id'] += max(self.user_mapping['remap_id']) + 1
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].apply(np.array).to_dict()
        self.test_user_dict = self.test_df.groupby('user_id')['asin'].apply(np.array).to_dict()

    def _get_numbers(self):
        self.items = set(self.kg_df_text['asin'].unique()) | set(self.train_df['asin'].unique()) | set(self.test_df['asin'].unique())
        self.users = set(self.train_df['user_id'].unique()) | set(self.test_df['user_id'].unique())
        self.entities = self.items | self.users
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        self.n_entities = len(self.entities)
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.users = list(self.users)

    def _print_info(self):
        self.logger.info(f'n_users:      {self.n_users:-7}')
        self.logger.info(f'n_items:      {self.n_items:-7}')
        self.logger.info(f'n_entities:   {self.n_entities:-7}')
        self.logger.info(f'n_train:      {self.n_train:-7}')
        self.logger.info(f'n_test:       {self.n_test:-7}')

    def _construct_embeddings(self):
        ''' randomly initialize all entity embeddings, we will overwrite the item embeddings next '''
        self.entity_embeddings = nn.Embedding(self.n_entities, self.args.embed_size)

        ''' construct text representations for items and embed them with BERT '''
        self.item_text_dict = {}
        for asin, group in self.kg_df_text.groupby('asin'):
            vals = group[['relation', 'attribute']].values
            self.item_text_dict[asin] = f' {self.args.sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping['text'] = self.item_mapping.apply(lambda x: self.item_text_dict[x['org_id']], axis=1)
        with torch.no_grad():
            self.entity_embeddings.weight[self.item_mapping['remap_id']] = embed_text(self.item_mapping['text'].to_list(), *init_bert(self.args))

    def _construct_graph(self):
        ''' create bipartite graph with initial vectors '''
        self.graph = dgl.heterograph({('user', 'bought', 'item'): (self.train_df['user_id'].values, self.train_df['asin'].values),
                                      ('item', 'bought_by', 'user'): (self.train_df['asin'].values, self.train_df['user_id'].values)})
        user_ids = torch.LongTensor(self.user_mapping['remap_id'])
        item_ids = torch.LongTensor(self.item_mapping['remap_id'])
        self.graph.ndata['e'] = {'user': self.entity_embeddings(user_ids),
                                 'item': self.entity_embeddings(item_ids)}
        self.graph.ndata['id'] = {'user': user_ids.to(self.device),
                                  'item': item_ids.to(self.device)}

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
