import torch
import torch.nn.functional as F

from .base_model import BaseModel


class TextBaseModel(BaseModel):
    ''' models that use textual semantic, text-based loss in addition to BPR '''

    def _copy_args(self, args):
        super()._copy_args(args)
        # weights specify which functions to use for semantic loss
        self.weight_key, self.distance_key = args.weight.split('_')

    def _add_vars(self, args):
        super()._add_vars(args)
        self.dist_fn = {
            'euclid': F.pairwise_distance,
            'cosine_minus': lambda x, y: -F.cosine_similarity(x, y),
        }[args.dist_fn]

    def bpr_loss(self, users, pos, negs):
        # todo: could probably disjoin the bpr loss from semantic loss to be cleaner
        users_emb, item_emb = self.representation
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
        return self.dist_fn(self.embedding_item(pos),
                            self.embedding_item(neg)
                            ).to(self.device)

    def bert_dist(self, pos, neg, users):
        ''' calculate similarity between textual representations of the sampled items '''
        return self.dist_fn(self.get_pos_items_reprs(pos, users),
                            self.get_neg_items_reprs(neg, users)
                            ).to(self.device)

    def get_pos_items_reprs(self, *args):
        ''' how do we represent positive items from sampled triplets '''
        raise NotImplementedError

    def get_neg_items_reprs(self, *args):
        ''' how do we represent negative items from sampled triplets '''
        raise NotImplementedError
