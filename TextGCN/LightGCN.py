import torch
from torch import nn
from torch.nn import functional as F

from .BaseModel import BaseModel


class LightGCN(BaseModel):

    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self._init_embeddings(params.emb_size, params.freeze)

    def _copy_params(self, params):
        super()._copy_params(params)
        self.dropout = params.dropout
        self.emb_size = params.emb_size
        self.n_layers = params.n_layers
        self.reg_lambda = params.reg_lambda
        if params.single:
            self.layer_combination = self.layer_combination_single

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.norm_matrix = dataset.norm_matrix

    def _add_vars(self, *args, **kwargs):
        super()._add_vars(*args, **kwargs)
        self.activation = F.selu  # F.softmax

    def _init_embeddings(self, emb_size, freeze):
        ''' randomly initialize entity embeddings '''
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=emb_size).to(self.device)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=emb_size).to(self.device)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        self.embedding_user.requires_grad_(not freeze)
        self.embedding_item.requires_grad_(not freeze)

    def score_pairwise(self, users_emb, items_emb, *args, **kwargs):
        return torch.sum(users_emb * items_emb, dim=1)

    def score_batchwise(self, users_emb, items_emb, *args, **kwargs):
        return torch.matmul(users_emb, items_emb.t())

    def layer_combination_single(self, vectors):
        '''
        only return the last layer representation
        instead of combining all layers
        '''
        return vectors[-1]

    def get_loss(self, data):
        users, pos, *negs = data.to(self.device).t()
        bpr_loss = self.bpr_loss(users, pos, negs)
        reg_loss = self.reg_loss(users, pos, negs)
        self._loss_values['bpr'] += bpr_loss
        self._loss_values['reg'] += reg_loss
        return bpr_loss + reg_loss

    def bpr_loss(self, users, pos, negs):
        ''' Bayesian Personalized Ranking pairwise loss '''
        users_emb, items_emb = self.representation
        users_emb = users_emb[users]
        pos_scores = self.score_pairwise(users_emb, items_emb[pos], users, pos)
        loss = 0
        for neg in negs:  # todo: vectorize
            neg_scores = self.score_pairwise(users_emb, items_emb[neg], users, neg)
            loss += torch.mean(self.activation(neg_scores - pos_scores))
        return loss / len(negs)

    def reg_loss(self, users, pos, negs):
        ''' regularization L2 loss '''
        loss = (
            self.embedding_user(users).norm(2).pow(2)
            + self.embedding_item(pos).norm(2).pow(2)
            + self.embedding_item(torch.stack(negs)).norm(2).pow(2).mean()
        )
        return self.reg_lambda * loss / len(users) / 2

    @property
    def embedding_matrix(self):
        ''' 0th layer embedding matrix '''
        return torch.cat([self.embedding_user.weight, self.embedding_item.weight])

    @property
    def _dropout_norm_matrix(self):
        ''' drop elements from adj table to help with overfitting '''
        indices = self.norm_matrix._indices()
        values = self.norm_matrix._values()
        mask = (torch.rand(len(values)) < (1 - self.dropout)).to(self.device)
        indices = indices[:, mask]
        values = values[mask] / (1 - self.dropout)
        matrix = torch.sparse_coo_tensor(indices, values, self.norm_matrix.size())
        return matrix.coalesce().to(self.device)

    @property
    def representation(self):
        '''
        aggregate embeddings from neighbors for each layer,
        combine layers into final representations
        '''
        norm_matrix = self._dropout_norm_matrix if self.training else self.norm_matrix
        current_lvl_emb_matrix = self.embedding_matrix
        node_embed_cache = [current_lvl_emb_matrix]
        for _ in range(self.n_layers):
            current_lvl_emb_matrix = self.layer_aggregation(norm_matrix, current_lvl_emb_matrix)
            node_embed_cache.append(current_lvl_emb_matrix)
        aggregated_embeddings = self.layer_combination(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def layer_aggregation(self, norm_matrix, emb_matrix):
        '''
        aggregate the neighbor's representations
        to get next layer node representation.

        default: normalized sum
        '''
        return torch.sparse.mm(norm_matrix, emb_matrix)

    def layer_combination(self, vectors):
        '''
        combine embeddings from all layers
        into final representation matrix.

        default: mean of all layers
        '''
        return torch.mean(torch.stack(vectors), axis=0)
