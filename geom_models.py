from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv
from base_model import BaseModel
import torch


class TorchGeometric(BaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        self.aggr = args.aggr
        self._build_layers(args.emb_size)

    def _build_layers(self, emb_size):
        LayerClass = {'gat': GATConv,
                      'gatv2': GATv2Conv,
                      'gcn': GCNConv,
                      'graphsage': SAGEConv}[self.model_name]
        self.layers = [LayerClass(emb_size, emb_size, aggr=self.aggr).to(self.device) for i in range(self.n_layers)]

    @property
    def representation(self):
        if self.training:
            norm_matrix = self._dropout_norm_matrix
        else:
            norm_matrix = self.norm_matrix

        curent_lvl_emb_matrix = self.embedding_matrix
        node_embed_cache = [curent_lvl_emb_matrix]

        edge_index = torch.cat([norm_matrix.indices(),
                                norm_matrix.indices().flip(dims=(0, 1))], axis=1)

        for layer in self.layers:
            curent_lvl_emb_matrix = layer(curent_lvl_emb_matrix, edge_index)
            node_embed_cache.append(curent_lvl_emb_matrix)

        aggregated_embeddings = self.layer_combination(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def layer_combination(self, vectors):
        return torch.mean(torch.stack(vectors, dim=1), dim=1)
        # return vectors[-1]
        # return F.log_softmax(vectors[-1], dim=1)
