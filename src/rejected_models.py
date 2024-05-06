import sys
from abc import ABC, abstractmethod

sys.path.append('..')
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    LGConv,
    SAGEConv,
    AntiSymmetricConv, DirGNNConv,
)
from transformers import set_seed

from .DatasetRanking import DatasetRanking
from .DatasetScoring import DatasetScoring
from .LightGCN import LightGCN
from .parsing import parse_args
from .RankingModel import RankingModel
from .ScoringModel import ScoringModel

'''
Models that I have tried but rejected because of poor performance
They are not maintained and might break in the future
'''


class TorchGeometric(LightGCN):
    ''' models based on the layers from torch_geometric library '''
    # requires the following parameters in args:
    #   aggr/aggregator: (mean, max, add)

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._build_layers(config.emb_size, config.model, config.aggr)
        print(self.layers)

    def _build_layers(self, emb_size, model_name, aggr):
        LayerClass = {
            'gat': GATConv,
            'gatv2': GATv2Conv,
            'gcn': GCNConv,
            'sage': SAGEConv,
            'lightgcn': LGConv,
            'antisymmetric': AntiSymmetricConv,
            'dirgcn': DirGNNConv,
        }[model_name]
        if model_name == 'lightgcn':
            self.layers = [LayerClass().to(self.device) for _ in range(self.n_layers)]
        elif model_name == 'antisymmetric':
            self.layers = [LayerClass(in_channels=emb_size).to(self.device) for _ in range(self.n_layers)]
        # elif model_name == 'dirgcn':
        #     self.layers = [LayerClass(conv=LGConv()).to(self.device) for _ in range(self.n_layers)]
        else:
            self.layers = [LayerClass(in_channels=emb_size, out_channels=emb_size, aggr=aggr).to(self.device) for _ in range(self.n_layers)]

    def forward(self):
        ''' same as in LightGCN, but call each layer instead of aggregating '''
        norm_matrix = self._dropout_norm_matrix if self.training else self.norm_matrix
        edge_index = torch.cat([norm_matrix.indices(), norm_matrix.indices().flip(dims=(0, 1))], axis=1)
        current_layer_emb_matrix = self.embedding_matrix
        node_embed_cache = [current_layer_emb_matrix]
        for layer in self.layers:
            current_layer_emb_matrix = layer(current_layer_emb_matrix, edge_index)
            node_embed_cache.append(current_layer_emb_matrix)
        aggregated_embeddings = self.layer_combination(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])


class LTRCosine(LightGCN):
    '''
    train the LightGCN model from scratch
    concatenate LightGCN vectors with text during training
    '''

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.users_text_repr = self.users_as_avg_reviews
        self.items_text_repr = {
            'ltr_reviews': self.items_as_avg_reviews,
            'ltr_kg': self.items_as_desc,
        }[config.model]

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


class LTRSimple(LightGCN):
    '''
    uses pretrained LightGCN model:
    concatenates LightGCN vectors with text during inference
    '''

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
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


class TextLightGCN(LightGCN, ABC):
    ''' models that use textual semantic, text-based loss in addition to BPR '''
    # requires the following parameters in args:
    #   weight: formula for semantic loss
    #   dist_fn: distance metric used in textual loss (euclid, cosine_minus)
    #   pos: how to represent the positive items from the sampled triplets (user, avg, kg)
    #   neg: how to represent the negative items from the sampled triplets (avg, kg)

    def _copy_params(self, config):
        super()._copy_params(config)
        # weights specify which functions to use for semantic loss
        self.weight_key, self.distance_key = config.weight.split('_')

    def _add_vars(self, config):
        super()._add_vars(config)
        self.dist_fn = {
            'euclid': F.pairwise_distance,
            'cosine_minus': lambda x, y: -F.cosine_similarity(x, y),
        }[config.dist_fn]

    def bpr_loss(self, users, pos, negs):
        # todo: could probably disjoin the bpr loss from semantic loss to be cleaner
        users_emb, item_emb = self.forward()
        users_emb = users_emb[users]
        pos_scores = self.score_pairwise(users_emb, item_emb[pos])
        loss = 0
        for neg in negs:
            neg_scores = self.score_pairwise(users_emb, item_emb[neg])
            bpr_loss = torch.mean(F.selu(neg_scores - pos_scores)) / len(negs)
            sem_loss = self.semantic_loss(users, pos, neg, pos_scores, neg_scores) / len(negs)
            self._loss_values['bpr'] += bpr_loss
            self._loss_values['sem'] += sem_loss
            loss += bpr_loss + sem_loss
        return loss

    def semantic_loss(self, users, pos, neg, pos_scores, neg_scores):
        ''' get semantic regularization using textual embeddings '''

        b = self.bert_dist(pos, neg, users)
        g = self.gnn_dist(pos, neg)

        distance = {
            'max(b-g)': F.relu(b - g),
            'max(g-b)': F.relu(g - b),
            '(b-g)': b - g,
            '(g-b)': g - b,
            '|b-g|': torch.abs(b - g),
            '|g-b|': torch.abs(g - b),
            'selu(g-b)': F.selu(g - b),
            'selu(b-g)': F.selu(b - g),
        }[self.distance_key]

        weight = {
            'max(p-n)': F.relu(pos_scores - neg_scores),
            '|p-n|': torch.abs(pos_scores - neg_scores),
            '(p-n)': pos_scores - neg_scores,
            '1': 1,
            '0': 0,  # in case we don't want semantic loss
        }[self.weight_key]

        return (weight * distance).mean()

    def gnn_dist(self, pos, neg):
        ''' calculate similarity between gnn representations of the sampled items '''
        return self.dist_fn(
            self.embedding_item(pos),
            self.embedding_item(neg),
        ).to(self.device)

    def bert_dist(self, pos, neg, users):
        ''' calculate similarity between textual representations of the sampled items '''
        return self.dist_fn(
            self.get_pos_items_reprs(pos, users),
            self.get_neg_items_reprs(neg, users),
        ).to(self.device)

    @abstractmethod
    def get_pos_items_reprs(self, *args, **kwargs):
        ''' how do we represent positive items from sampled triplets '''

    @abstractmethod
    def get_neg_items_reprs(self, *args, **kwargs):
        ''' how do we represent negative items from sampled triplets '''


class TextModelKG(TextLightGCN):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        ''' all items are represented with their descriptions '''

        if config.pos == 'kg' or config.model == 'kg':
            self.get_pos_items_reprs = self.get_item_desc
        if config.neg == 'kg' or config.model == 'kg':
            self.get_neg_items_reprs = self.get_item_desc

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.items_as_desc = dataset.items_as_desc

    def get_item_desc(self, items):
        return self.items_as_desc[items]


class TextModelReviews(TextLightGCN):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        ''' how do we textually represent items in sampled triplets '''
        if config.pos == 'avg' or config.model == 'reviews':
            self.get_pos_items_reprs = self.get_item_reviews_mean
        elif config.pos == 'user':
            self.get_pos_items_reprs = self.get_item_reviews_user

        if config.neg == 'avg' or config.model == 'reviews':
            self.get_neg_items_reprs = self.get_item_reviews_mean

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.reviews_vectors = dataset.reviews_vectors
        self.items_as_avg_reviews = dataset.items_as_avg_reviews

    def get_item_reviews_mean(self, items):
        ''' represent items with mean of their reviews '''
        return self.items_as_avg_reviews[items]

    def get_item_reviews_user(self, items, users):
        ''' represent items with the review of corresponding user '''
        df = self.reviews_vectors.loc[torch.stack([items, users], axis=1).tolist()]
        return torch.from_numpy(df.values).to(self.device)


class TextModel(TextModelReviews, TextModelKG):
    pass


class TestModel(TextModel):
    ''' evaluate the Simple models (concat text emb at inference) '''

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.users_as_avg_desc = dataset.users_as_avg_desc
        self.items_as_desc = dataset.items_as_desc

        for fn in [self.representation_rev_rev,
                   self.representation_kg_kg,
                   self.representation_rev_kg,
                   self.representation_kg_rev]:
            self.forward = fn
            self.evaluate()
        exit()

    def representation_kg_kg(self):
        return self.users_as_avg_desc, self.items_as_desc

    def representation_rev_kg(self):
        return self.users_as_avg_reviews, self.items_as_desc

    def representation_kg_rev(self):
        return self.users_as_avg_desc, self.items_as_avg_reviews

    def representation_rev_rev(self):
        return self.users_as_avg_reviews, self.items_as_avg_reviews


class TorchGeometricRank(TorchGeometric, RankingModel):
    ...


class TorchGeometricScore(TorchGeometric, ScoringModel):
    ...


def main():
    params = parse_args()
    set_seed(params.seed)

    # params.aggr = 'add'
    params.aggr = 'max'
    # params.aggr = 'mean'

    # dataset = DatasetRanking(params)
    # model = TorchGeometricRank(params, dataset)

    dataset = DatasetScoring(params)
    model = TorchGeometricScore(params, dataset)

    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    model.fit(loader)


if __name__ == '__main__':
    main()
