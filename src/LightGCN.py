import torch
from torch import nn

from .BaseModel import BaseModel
from .RankingModel import RankingModel
from .ScoringModel import ScoringModel


class LightGCN(BaseModel):
    '''
    LightGCN model from https://arxiv.org/pdf/2002.02126.pdf
    without an objective function
    uses only user-item interactions
    '''

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_embeddings(config.emb_size, config.freeze)

    def _copy_params(self, config):
        super()._copy_params(config)
        self.dropout = config.dropout
        self.emb_size = config.emb_size
        self.n_layers = config.n_layers
        if config.single:
            self.layer_combination = self.layer_combination_single

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.norm_matrix = dataset.norm_matrix

    def _init_embeddings(self, emb_size, freeze):
        ''' randomly initialize entity embeddings '''
        self.embedding_user = nn.Embedding(
            num_embeddings=self.n_users,
            embedding_dim=emb_size,
            device=self.device,
            _freeze=freeze,
        )
        self.embedding_item = nn.Embedding(
            num_embeddings=self.n_items,
            embedding_dim=emb_size,
            device=self.device,
            _freeze=freeze,
        )
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def score_pairwise(self, users_emb, items_emb, *args, **kwargs):
        return torch.sum(users_emb * items_emb, dim=1)

    def score_batchwise(self, users_emb, items_emb, *args, **kwargs):
        return torch.matmul(users_emb, items_emb.T)

    def layer_combination_single(self, vectors):
        '''
        only return the last layer representation
        instead of combining all layers
        '''
        return vectors[-1]

    @property
    def embedding_matrix(self):
        ''' 0th layer embedding matrix '''
        return torch.cat([self.embedding_user.weight, self.embedding_item.weight])

    @property
    def _dropout_norm_matrix(self):
        ''' drop elements from adj table to help with overfitting '''
        indices = self.norm_matrix._indices()
        values = self.norm_matrix._values()

        mask = (torch.rand(len(values), device=self.device) < (1 - self.dropout))
        indices = indices[:, mask]
        values = values[mask] / (1 - self.dropout)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=self.norm_matrix.size(),
            device=self.device,
            is_coalesced=True,
        )

    def forward(self):
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

    def reg_loss(self, users, items):
        ''' regularization L2 loss '''
        loss = self.embedding_user(users).norm(2).pow(2) + self.embedding_item(items).norm(2).pow(2)
        return loss / (len(users) + len(items))

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


class LightGCNRank(LightGCN, RankingModel):
    '''
    Ranking version of LightGCN
    same objective as in original paper
    '''


class LightGCNScore(LightGCN, ScoringModel):
    '''
    Scoring version LightGCN
    different objective than the original paper
    '''

    def _init_embeddings(self, *args, **kwargs):
        super()._init_embeddings(*args, **kwargs)
        self.embedding_user.max_norm = 1.0
        self.embedding_item.max_norm = 1.0
