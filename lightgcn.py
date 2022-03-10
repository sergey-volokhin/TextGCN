import torch

from base_model import BaseModel


class LightGCN(BaseModel):

    def get_representation(self):
        norm_matrix = self._dropout_norm_matrix()
        h = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        node_embed_cache = [h]
        print(h.shape)
        for _ in range(self.n_layers):
            h = torch.sparse.mm(norm_matrix, h)
            print(h.shape)
            node_embed_cache.append(h)
        exit()
        node_embed_cache = self.aggregate_layers(node_embed_cache)
        return torch.split(node_embed_cache, [self.n_users, self.n_items])

    def aggregate_layers(self, vectors):
        return torch.mean(torch.stack(vectors, dim=1), dim=1)
