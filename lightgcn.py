import torch

from base_model import BaseModel


class LightGCN(BaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        self.load_model()

        self.to(self.device)
        self._build_optimizer()
        self.logger.info(self)

    def _dropout_graph(self):
        index = self.adj_matrix.indices().t()
        values = self.adj_matrix.values()
        random_index = (torch.rand(len(values)) + self.keep_prob).int().bool()
        index = index[random_index]
        values = values[random_index] / self.keep_prob
        return torch.sparse.FloatTensor(index.t(), values, self.adj_matrix.size())

    def get_representation(self):
        ''' calculating the node representation using all the layers'''
        g_droped = self._dropout_graph()

        h = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        node_embed_cache = [h]
        for _ in range(self.n_layers):
            h = torch.sparse.mm(g_droped, h)
            node_embed_cache.append(h)

        node_embed_cache = torch.mean(torch.stack(node_embed_cache, dim=1), dim=1)
        return torch.split(node_embed_cache, [self.n_users, self.n_items])
