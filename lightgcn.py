import torch
from torch import nn
from torch.nn import functional as F

from base_model import BaseModel


class LightGCN(BaseModel):

    def layer_propagate(self, norm_matrix, emb_matrix):
        return torch.sparse.mm(norm_matrix, emb_matrix)

    def layer_aggregation(self, vectors):
        return torch.mean(torch.stack(vectors, dim=1), dim=1)

    def embedding_matrix(self):
        ''' get the embedding matrix of 0th layer '''
        return torch.cat([self.embedding_user.weight, self.embedding_item.weight])


class LightGCNAttn(LightGCN):

    def _add_vars(self):
        super()._add_vars()

        # NGCF weights
        self.W_ngcf = nn.Parameter(torch.empty((2, 1)), requires_grad=True)
        nn.init.xavier_normal_(self.W_ngcf, gain=nn.init.calculate_gain('relu'))

    def layer_propagate(self, norm_matrix, emb_matrix):
        '''
            propagate messages through layer using NGCF attention
                                         e_i                   e_u ⊙ e_i
            e_u = σ(W1*e_u + W1*SUM---------------- + W2*SUM----------------)
                                   sqrt(|N_u||N_i|)         sqrt(|N_u||N_i|)
        '''

        summ = torch.sparse.mm(norm_matrix, emb_matrix)
        return F.leaky_relu((self.W_ngcf[0] * emb_matrix) +
                            (self.W_ngcf[0] * summ) +
                            (self.W_ngcf[1] * torch.mul(emb_matrix, summ)))


class LightGCNWeight(LightGCN):

    def _add_vars(self):
        super()._add_vars()

        # linear combination of layer represenatations
        self.W_layers = nn.Parameter(torch.empty((self.n_layers + 1, 1)), requires_grad=True)
        nn.init.xavier_normal_(self.W_layers, gain=nn.init.calculate_gain('relu'))

    def layer_aggregation(self, vectors):
        ''' aggregate layer representations into one vector '''
        return (self.W_layers.T * torch.stack(vectors).T).T.sum(axis=0)  # TODO could improve. attention?


class LightGCNSingle(LightGCN):
    def layer_aggregation(self, vectors):
        return vectors[-1]


class LightGCNSingleAttn(LightGCNSingle, LightGCNAttn):
    pass
