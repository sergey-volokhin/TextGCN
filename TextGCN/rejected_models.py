import torch
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, LGConv, SAGEConv

from .base_model import BaseModel


class TorchGeometric(BaseModel):
    ''' models based on the layers from torch_geometric library '''

    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self._build_layers(params.emb_size, params.model, params.aggr)

    def _build_layers(self, emb_size, model_name, aggr):
        LayerClass = {
            'gat': GATConv,
            'gatv2': GATv2Conv,
            'gcn': GCNConv,
            'graphsage': SAGEConv,
            'lightgcn': LGConv,
        }[model_name]
        if model_name == 'lightgcn':
            self.layers = [LayerClass().to(self.device) for _ in range(self.n_layers)]
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

    def layer_combination(self, vectors):  # todo: why is this here? isn't basemodel the same
        return torch.mean(torch.stack(vectors), axis=0)


class LTRCosine(BaseModel):
    '''
    train the LightGCN model from scratch
    concatenate LightGCN vectors with text during training
    '''

    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.users_text_repr = self.users_as_avg_reviews
        self.items_text_repr = {
            'ltr_reviews': self.items_as_avg_reviews,
            'ltr_kg': self.items_as_desc,
        }[params.model]

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def score_pairwise(self, users_emb, item_emb, users, items):
        user_part = torch.cat([users_emb, self.users_text_repr[users]], axis=1)
        item_part = torch.cat([item_emb, self.items_text_repr[items]], axis=1)
        # return F.cosine_similarity(user_part, item_part)
        return torch.sum(torch.mul(user_part, item_part), dim=1)

    def score_batchwise(self, users_emb, items_emb, users):
        user_part = torch.cat([users_emb, self.users_text_repr[users]], axis=1)
        item_part = torch.cat([items_emb, self.items_text_repr], axis=1)
        return torch.matmul(user_part, item_part.t())


class LTRSimple(BaseModel):
    '''
    uses pretrained LightGCN model:
    concatenates LightGCN vectors with text during inference
    '''

    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.users_text_repr = self.users_as_avg_reviews

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.reviews_vectors = dataset.reviews_vectors
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def score_batchwise_ltr(self, users_emb, items_emb, users):
        user_part = torch.cat([users_emb, self.users_text_repr[users]], axis=1)
        item_part = torch.cat([items_emb, self.items_text_repr[range(self.n_items)]], axis=1)
        return torch.matmul(user_part, item_part.t())

    def fit(self, batches):
        ''' no actual training happens, we use pretrained model '''
        self.score_batchwise = self.score_batchwise_ltr

        self.logger.info('Performance when using pos=avg:')
        self.items_text_repr = self.items_as_avg_reviews
        self.evaluate()

        self.logger.info('Performance when using pos=kg:')
        self.items_text_repr = self.items_as_desc
        self.evaluate()
