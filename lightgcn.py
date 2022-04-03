import torch
from torch import nn
from torch.nn import functional as F

from base_model import BaseModel


class LightGCN(BaseModel):

    @property
    def representation(self):
        '''
            calculate current embeddings,
            pull the through layers and aggregate
        '''
        h = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        norm_matrix = self._dropout_norm_matrix
        node_embed_cache = [h]
        for _ in range(self.n_layers):
            h = self.layer_propagate(norm_matrix, h)
            node_embed_cache.append(h)
        aggregated_embeddings = self.layer_aggregation(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def layer_propagate(self, norm_matrix, h):
        return torch.sparse.mm(norm_matrix, h)

    def layer_aggregation(self, vectors):
        return torch.mean(torch.stack(vectors, dim=1), dim=1)


class LightGCNAttn(LightGCN):

    def _add_torch_vars(self):
        # ngcf variables init
        self.W_ngcf = nn.Parameter(torch.empty((2, 1)), requires_grad=True)
        nn.init.xavier_normal_(self.W_ngcf, gain=nn.init.calculate_gain('relu'))

    def layer_propagate(self, norm_matrix, h):
        '''
            propagate messages through layer using ngcf formula
                                         e_i                   e_u ⊙ e_i
            e_u = σ(W1*e_u + W1*SUM---------------- + W2*SUM----------------)
                                   sqrt(|N_u||N_i|)         sqrt(|N_u||N_i|)
        '''

        summ = torch.sparse.mm(norm_matrix, h)
        return F.leaky_relu((self.W_ngcf[0] * h) + (self.W_ngcf[0] * summ) +
                            (self.W_ngcf[1] * torch.mul(h, summ)))


class LightGCNWeight(LightGCN):

    def _add_torch_vars(self):
        # linear combination of layer represenatations
        self.W_layers = nn.Parameter(torch.empty((self.n_layers + 1, 1)), requires_grad=True)
        nn.init.xavier_normal_(self.W_layers, gain=nn.init.calculate_gain('relu'))

    def layer_aggregation(self, vectors):
        ''' aggregate layer representations into one vector '''
        return (self.W_layers.T * torch.stack(vectors).T).T.sum(axis=0)  # TODO could improve. attention?


class LightGCNSingle(LightGCN):
    def layer_aggregation(self, vectors):
        return vectors[-1]


class LightSingleGCNAttn(LightGCNSingle, LightGCNAttn):
    pass
