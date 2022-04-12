import torch

from base_model import BaseModel


class LightGCN(BaseModel):

    def layer_propagate(self, norm_matrix, emb_matrix):
        return torch.sparse.mm(norm_matrix, emb_matrix)

    def layer_aggregation(self, vectors):
        return torch.mean(torch.stack(vectors, dim=1), dim=1)
