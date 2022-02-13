import dgl
import os
import random
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from cppimport import imp_from_filepath
from tqdm import tqdm

from utils import embed_text, minibatch, shuffle


class DataLoader:

    def __init__(self, args):
        self._copy_args(args)
        self._load_files()
        self._print_info()
        self._build_dicts()
        self._init_embeddings()
        self._construct_graph()

    def _copy_args(self, args):
        self.seed = args.seed
        self.logger = args.logger
        self.device = args.device
        self.path = args.datapath
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size

    def _load_files(self):
        self.logger.info('loading data')
        self.train_df = pd.read_table(self.path + 'train.tsv', dtype=np.int32, header=0, names=['user_id', 'asin'])
        self.test_df = pd.read_table(self.path + 'test.tsv', dtype=np.int32, header=0, names=['user_id', 'asin'])

        ''' load C++ sampling '''
        sys.path.append('sources')
        self.c_sampler = imp_from_filepath(os.path.join(os.path.dirname(__file__), "sources/sampling.cpp"))
        self.c_sampler.seed(self.seed)

    def _init_embeddings(self):
        ''' randomly initialize entity embeddings '''
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.emb_size, device=self.device)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.emb_size, device=self.device)
        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)

    def _build_dicts(self):
        self.train_user_dict = self.train_df.groupby('user_id')['asin'].apply(sorted).to_dict()
        self.test_user_dict = self.test_df.groupby('user_id')['asin'].apply(sorted).to_dict()
        self.positive_lists = [self.train_user_dict[u] for u in list(range(self.n_users))]

    def _print_info(self):
        self.n_users = self.train_df.user_id.nunique()
        self.n_items = self.train_df.asin.nunique()
        self.n_train = self.train_df.shape[0]
        self.n_test = self.test_df.shape[0]
        self.n_batches = (self.n_train - 1) // self.batch_size + 1

        self.logger.info(f"n_users:    {self.n_users:-7}")
        self.logger.info(f"n_items:    {self.n_items:-7}")
        self.logger.info(f"n_train:    {self.n_train:-7}")
        self.logger.info(f"n_test:     {self.n_test:-7}")
        self.logger.info(f"n_batches:  {self.n_batches:-7}")

    def sampler(self, neg_ratio=1):
        '''
            sample n_train users with their positive and negative items
            S: a (n_train, 3) shaped np.array of triples (user, pos, neg)
            yields 3 batched lists: [user1, ...], [pos1, ...], [neg1, ...]
        '''
        try:
            S = self.c_sampler.sample_negative(self.n_users,
                                               self.n_items,
                                               self.n_train,
                                               self.positive_lists,
                                               neg_ratio)
        except Exception:
            S = self.python_sampler()

        users = torch.Tensor(S[:, 0]).long().to(self.device)
        pos_items = torch.Tensor(S[:, 1]).long().to(self.device)
        neg_items = torch.Tensor(S[:, 2]).long().to(self.device)
        return minibatch(*shuffle(users, pos_items, neg_items), batch_size=self.batch_size)

    def python_sampler(self):
        ''' fallback python sampling: 100x slower than C++ '''
        self.logger.warn('C++ sampling not available, falling back to slow python')
        users = np.random.randint(0, self.n_users, self.n_train)
        result = []
        for user in users:
            pos_for_user = self.positive_lists[user]
            pos = np.random.choice(pos_for_user, 1)
            while True:
                neg = random.randint(0, self.n_items)
                if neg not in pos_for_user:
                    break
            result.append([user, pos, neg])
        return np.array(result)

    def get_user_pos_items(self, users):
        ''' returns positive items per batch of users '''
        return [self.train_user_dict[user] for user in users]

    def _construct_graphs(self):
        raise NotImplementedError


class DataLoaderText(DataLoader):

    def __init__(self, args):
        super().__init__(args)
        self._set_single_vector()

    def _copy_args(self, args):
        super()._copy_args(args)

        self.sep = args.sep
        self.bert_model = args.bert_model
        self.single_vector = args.single_vector
        self.emb_batch_size = args.emb_batch_size

    def _load_files(self):
        super()._load_files()
        self.kg_df_text = pd.read_table(self.path + 'kg_readable.tsv', dtype=str)[['asin', 'relation', 'attribute']]
        self.user_mapping = pd.read_csv(self.path + 'user_list.txt', sep=' ')[['org_id', 'remap_id']]
        self.item_mapping = pd.read_csv(self.path + 'item_list.txt', sep=' ')[['org_id', 'remap_id']]

    def _construct_text_representation(self):
        self.item_text_dict = {}
        for asin, group in tqdm(self.kg_df_text.groupby('asin'), desc='construct text repr', dynamic_ncols=True):
            vals = group[['relation', 'attribute']].values
            self.item_text_dict[asin] = f' {self.sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping['text'] = self.item_mapping['org_id'].map(self.item_text_dict)

    def _init_embeddings(self):
        ''' construct BERT representation for items and overwrite model item embeddings with them'''

        super()._init_embeddings()

        emb_path = f'embeddings_{self.bert_model.split("/")[-1]}.torch'
        if not os.path.exists(self.path + emb_path):
            self._construct_text_representation()
            embeddings = embed_text(self.item_mapping['text'],
                                    self.path,
                                    self.bert_model,
                                    self.emb_batch_size,
                                    self.device)
        else:
            embeddings = torch.load(self.path + emb_path, map_location=self.device)

        with torch.no_grad():
            self.embedding_item.weight[:] = torch.tensor(embeddings).to(self.device)

    def _set_single_vector(self):
        ''' set up the initial user vector if using single-vector-initiation '''
        if self.single_vector:
            self.user_vector = torch.nn.parameter.Parameter(torch.Tensor(self.emb_size))
            torch.nn.init.xavier_uniform_(self.user_vector.unsqueeze(1))
        else:
            self.user_vector = None

    def _construct_graph(self):
        ''' create bipartite graph with initial vectors '''
        self.graph = dgl.heterograph({('user', 'bought', 'item'): (self.train_df['user_id'].values, self.train_df['asin'].values),
                                      ('item', 'bought_by', 'user'): (self.train_df['asin'].values, self.train_df['user_id'].values)})
        self.graph = self.graph.to(self.device)
        user_ids = torch.tensor(self.user_mapping['remap_id'], dtype=torch.long, device=self.device)
        item_ids = torch.tensor(self.item_mapping['remap_id'], dtype=torch.long, device=self.device)
        self.graph.ndata['e'] = {'user': self.embedding_user(user_ids),
                                 'item': self.embedding_item(item_ids)}
        self.graph.ndata['id'] = {'user': user_ids,
                                  'item': item_ids}


class DataLoaderLightGCN(DataLoader):

    def _construct_graph(self):
        if os.path.exists(self.path + 's_pre_adj_mat.npz'):
            norm_adj = sp.load_npz(self.path + 's_pre_adj_mat.npz')
            self.logger.info("successfully loaded adjacency matrix")
        else:
            norm_adj = self._generate_adj_matrix()
        self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.graph = self.graph.coalesce().to(self.device)

    def _generate_adj_matrix(self):
        self.logger.info("generating adjacency matrix")
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32).tolil()
        user_item_net = sp.csr_matrix((np.ones(self.train_df.user_id.shape),
                                       (self.train_df.user_id, self.train_df.asin)),
                                      shape=(self.n_users, self.n_items))
        R = user_item_net.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
        self.logger.info('generated. saving...')
        sp.save_npz(self.path + 's_pre_adj_mat.npz', norm_adj)
        return norm_adj

    def _convert_sp_mat_to_sp_tensor(self, x):
        coo = x.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
