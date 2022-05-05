import torch
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, LGConv, SAGEConv

from base_model import BaseModel


class TorchGeometric(BaseModel):

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self._build_layers(args.emb_size, args.model, args.aggr)

    def _build_layers(self, emb_size, model_name, aggr):
        LayerClass = {'gat': GATConv,
                      'gatv2': GATv2Conv,
                      'gcn': GCNConv,
                      'graphsage': SAGEConv,
                      'lightgcn': LGConv,
                      }[model_name]
        if model_name == 'lightgcn':
            self.layers = [LayerClass().to(self.device) for i in range(self.n_layers)]
        else:
            self.layers = [LayerClass(emb_size, emb_size, aggr=aggr).to(self.device) for _ in range(self.n_layers)]

    @property
    def representation(self):
        norm_matrix = self._dropout_norm_matrix if self.training else self.norm_matrix
        edge_index = torch.cat([norm_matrix.indices(), norm_matrix.indices().flip(dims=(0, 1))], axis=1)
        current_layer_emb_matrix = self.embedding_matrix
        node_embed_cache = [current_layer_emb_matrix]

        for layer in self.layers:
            current_layer_emb_matrix = layer(current_layer_emb_matrix, edge_index)
            node_embed_cache.append(current_layer_emb_matrix)

        aggregated_embeddings = self.layer_combination(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def layer_combination(self, vectors):
        return torch.mean(torch.stack(vectors), axis=0)
