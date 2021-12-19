import collections
import os
import random

import dgl
import numpy as np
import pandas as pd
import torch

from utils import embed_text, init_bert


class DataLoader(object):
    def __init__(self, args, seed=1234):

        random.seed(seed)
        path = args.datapath
        self.device = args.device
        self.logger = args.logger
        self.logger.info('loading data')

        self.train_df = train_df = self._load_df(os.path.join(path, 'train.tsv')).sort_values(by=['user_id'])
        self.test_df = test_df = self._load_df(os.path.join(path, 'test.tsv')).sort_values(by=['user_id'])
        self.train_user_dict = self._convert_uv_pair2dict(train_df)
        self.test_user_dict = self._convert_uv_pair2dict(test_df)
        kg_df_text = pd.read_table(os.path.join(path, 'kg_readable.tsv'))
        self.n_users = len(set(train_df['user_id'].unique()) | set(test_df['user_id'].unique()))
        self.n_items = len(set(train_df['item_id'].unique()) | set(test_df['item_id'].unique()))
        self.n_train = train_df.shape[0]
        self.n_test = test_df.shape[0]
        self.item_id_range = np.arange(self.n_users, self.n_users + self.n_items)
        entities = set(kg_df_text['asin'].unique()) | \
            set(train_df['user_id'].unique()) | \
            set(train_df['item_id'].unique())
        self.n_entities = len(entities)
        self.print_info()

        ''' get initial text representation of items and embed it '''
        self.item_text_dict = {}
        for asin, group in kg_df_text.groupby('asin'):
            vals = group[['relation', 'attribute']].values
            self.item_text_dict[asin] = ' [SEP] '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping = pd.read_csv(os.path.join(path, '../item_list.txt'), sep=' ')
        self.item_mapping['text'] = self.item_mapping.apply(lambda x: self.item_text_dict[x['org_id']], axis=1)
        embedding = embed_text(self.item_mapping['text'].to_list(), *init_bert(args))

        ''' create the graph, put initial vectors for items and users '''
        self.graph = dgl.heterograph({('user', 'bought', 'item'): (self.train_df.values[:, 0], self.train_df.values[:, 1])})
        self.graph.nodes['item'].data['e'] = embedding
        # self.graph.nodes['user'].data['e'] = embedding

    # @property
    # def graph(self):
    #     graph = dgl.DGLGraph()
    #     graph.readonly()
    #     graph.ndata['id'] = torch.arange(self.n_entities, dtype=torch.long)
    #     graph.ndata['value'] = {'user': self.user_mapping['org_id'].values}
    #     graph.apply_nodes(lambda nodes: {'value': self.item_text_dict[nodes.data['id']]}, ntype='item')
    #     e_idxs = g.filter_nodes(lambda nodes: nodes.data['type'] == i)
    #     g.apply_edges(, e_idxs)
    #     return graph

    # @property
    # def train_g(self):
    #     g = dgl.DGLGraph()
    #     g.add_nodes(self.n_entities)
    #     g.add_edges(self.train_kg_triplet[:, 2], self.train_kg_triplet[:, 0])
    #     g.readonly()
    #     g.ndata['id'] = torch.arange(self.n_entities, dtype=torch.long)
    #     g.edata['type'] = torch.LongTensor(self.train_kg_triplet[:, 1])
    #     # g.edata['w'] = torch.LongTensor()
    #     # nn.init.xavier_uniform_(g.edata['w'], gain=nn.init.calculate_gain('relu'))
    #     return g

    def print_info(self):
        self.logger.info(f'n_users:      {self.n_users:-7}')
        self.logger.info(f'n_items:      {self.n_items:-7}')
        self.logger.info(f'n_entities:   {self.n_entities:-7}')
        self.logger.info(f'n_train:      {self.n_train:-7}')
        self.logger.info(f'n_test:       {self.n_test:-7}')

    def _load_df(self, file_name):
        if os.path.isfile(file_name):
            return pd.read_table(file_name, dtype=np.int32)
        self.logger.error('{} does not exit.'.format(file_name))

    def _convert_uv_pair2dict(self, uv_df):
        u_v_dict = {}
        for ind, g in uv_df.groupby('user_id'):
            u_v_dict[ind] = g['item_id'].values
        return u_v_dict

    def _generate_kg(self, n_entity, kg_triplet, add_etype=True):
        g = dgl.DGLGraph()
        g.add_nodes(n_entity)
        g.add_edges(kg_triplet[:, 2], kg_triplet[:, 0])
        g.readonly()
        g.ndata['id'] = torch.arange(n_entity, dtype=torch.long)
        if add_etype:
            g.edata['type'] = torch.LongTensor(kg_triplet[:, 1])
        return g

    def create_edge_sampler(self, graph, **kwargs):
        edge_sampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        return edge_sampler(graph,
                            neg_sample_size=1,
                            shuffle=True,
                            return_false_neg=True,
                            exclude_positive=False,
                            **kwargs)

    def sampler(self, batch_size, pos_mode, num_workers=8):
        if batch_size < 0 or batch_size > self.n_train:
            batch_size = self.n_train
            n_batch = 1
        else:
            n_batch = self.n_train // batch_size + 1

        if pos_mode == 'unique':
            exist_users = list(self.train_user_dict)
            i = 0
            while i < n_batch:
                i += 1
                if batch_size <= self.n_users:
                    users = random.sample(exist_users, batch_size)
                else:
                    users = random.choices(exist_users, k=batch_size)
                pos_items, neg_items = [], []
                for u in users:
                    pos_items.append(random.choice(self.train_user_dict[u]))
                    while True:
                        neg_i_id = random.randrange(self.n_items)
                        if neg_i_id not in self.train_user_dict[u]:
                            break
                    neg_items.append(neg_i_id)
                yield users, pos_items, neg_items, None
        else:
            for pos_g, neg_g in self.create_edge_sampler(self.cf_graph,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         negative_mode='head'):
                false_neg = neg_g.edata['false_neg']
                pos_g.copy_from_parent()
                neg_g.copy_from_parent()
                i_idx, u_idx = pos_g.all_edges(order='eid')
                neg_i_idx, _ = neg_g.all_edges(order='eid')
                users = torch.LongTensor(pos_g.ndata['id'][u_idx]).to(self.device)
                pos_items = torch.LongTensor(pos_g.ndata['id'][i_idx]).to(self.device)
                neg_items = torch.LongTensor(neg_g.ndata['id'][neg_i_idx]).to(self.device)
                yield users, pos_items, neg_items, false_neg
