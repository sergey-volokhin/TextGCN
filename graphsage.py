import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from base_model import BaseModel
from utils import profile


class GraphSageMean(BaseModel):

    @property
    def _dropout_norm_matrix(self):
        ''' don't normalize adj matrix by number of neighbors '''
        return super()._dropout_norm_matrix.ceil()

    def layer_aggregation(self, norm_matrix, emb_matrix):
        '''
            mean aggregation:
                node's vector equals mean of (neighbors ∪ itself)
        '''
        numerator = (torch.sparse.mm(norm_matrix, emb_matrix) + emb_matrix).T
        denominator = (torch.sparse.sum(norm_matrix, dim=1).to_dense() + 1).T
        return torch.sigmoid((numerator / denominator).T)

    def layer_combination(self, vectors):
        return F.normalize(vectors[-1])


class GraphSagePool(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.max_neighbors = args.max_neighbors

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.neighbors = dataset.neighbors
        self.lengths = [x.size(0) for x in self.neighbors]

    def _add_vars(self):
        super()._add_vars()
        self.zeros = torch.zeros(self.emb_size).unsqueeze(0).to(self.device)

    def _pad_neighbors(self, vector):
        ''' pad or cut the number of neighbors to the pre-selected number '''
        to_pad = self.max_neighbors - vector.size(0)
        if to_pad > 0:
            return F.pad(vector, (0, to_pad), value=-1)
        return vector[torch.randperm(vector.size(0))][:self.max_neighbors]

    def _init_embeddings(self, emb_size):
        super()._init_embeddings(emb_size)

        ''' fully connected layer'''
        self.dense = nn.Linear(emb_size, emb_size, device=self.device)
        # nn.init.xavier_normal_(self.dense, gain=nn.init.calculate_gain('relu'))

    def layer_aggregation(self, norm_matrix, emb_matrix):

        '''
            recalculate embedding matrix:
            pass through dense layer and apply non-linearity
            then get element-wise max over the neighbors
        '''
        transformed_emb = torch.sigmoid(self.dense(emb_matrix))
        emb_mat = torch.cat((transformed_emb, self.zeros))

        adj = torch.stack([self._pad_neighbors(x) for x in self.neighbors])  # <- _pad_neighbors
        return emb_mat[adj].max(axis=1).values

        ''' alternative max calculation '''  # <- torch._C._nn.pad_sequence take too long
        # split = emb_mat[torch.cat(self.neighbors)].split(self.lengths)
        # batch_size = 1000
        # batches = [split[j:j + batch_size] for j in range(0, len(split), batch_size)]
        # padded = []
        # for batch in batches:
        #     padded.append(nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=-999).max(axis=1).values)
        # return torch.cat(padded)

        # padded = nn.utils.rnn.pad_sequence(split, batch_first=True, padding_value=-999) # <- tries to pad all, up to 2000 neighbors
        # return padded.max(axis=1).values
        # return torch.stack([x.max(axis=0).values for x in split])

    def layer_combination(self, vectors):
        return F.normalize(vectors[-1])

    # @profile
    # @property
    def representation(self):
        '''
            get the users' and items' final representations
            aggregate embeddings from neighbors for each layer
            combine layers at the end
        '''
        if self.training:
            norm_matrix = self._dropout_norm_matrix
        else:
            norm_matrix = self.norm_matrix

        curent_lvl_emb_matrix = self.embedding_matrix
        node_embed_cache = [curent_lvl_emb_matrix]
        for _ in range(self.n_layers):
            curent_lvl_emb_matrix = self.layer_aggregation(norm_matrix, curent_lvl_emb_matrix)
            node_embed_cache.append(curent_lvl_emb_matrix)
        aggregated_embeddings = self.layer_combination(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])
