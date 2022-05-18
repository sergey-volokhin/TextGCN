import torch
import torch.nn.functional as F

from base_model import BaseModel


class TextBaseModel(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.weight = args.weight

    def _add_vars(self, args):
        super()._add_vars(args)
        self.dist_fn = {
            'euclid': F.pairwise_distance,
            'cosine_minus': lambda x, y: -F.cosine_similarity(x, y),
        }[args.dist_fn]

    def bpr_loss(self, users, pos, negs):
        users_emb, item_emb = self.representation
        users_emb = users_emb[users]
        pos_scores = self.score(users_emb, item_emb[pos], users, pos, 'pos')
        loss = 0
        for neg in negs:
            neg_scores = self.score(users_emb, item_emb[neg], users, neg, 'neg')
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
        distance = {'b': b,
                    'max(b)': F.relu(b),
                    'max(b-g)': F.relu(b - g),
                    'max(g-b)': F.relu(g - b),
                    '(b-g)': b - g,
                    '(g-b)': g - b,
                    '|b-g|': torch.abs(b - g),
                    '|g-b|': torch.abs(g - b),
                    'selu(g-b)': F.selu(g - b),
                    'selu(b-g)': F.selu(b - g),
                    }[self.weight.split('_')[1]]

        weight = {'max(p-n)': F.relu(pos_scores - neg_scores),
                  '|p-n|': torch.abs(pos_scores - neg_scores),
                  '(p-n)': pos_scores - neg_scores,
                  '1': 1,
                  '0': 0,  # in case we want to not have semantic loss
                  }[self.weight.split('_')[0]]

        return (weight * distance).mean()

    def gnn_dist(self, pos, neg):
        ''' calculate similarity between gnn representations of the sampled items '''
        cands = F.normalize(self.embedding_item(pos))
        refs = F.normalize(self.embedding_item(neg))
        return self.dist_fn(cands, refs).to(self.device)

    def bert_dist(self, pos, neg, users):
        ''' calculate similarity between textual representations of the sampled items '''
        cands = self.get_pos_items_reprs(pos, users)
        refs = self.get_neg_items_reprs(neg, users)
        return self.dist_fn(cands, refs).to(self.device)

    def get_pos_items_reprs(self, items, users=None):
        ''' how do we represent positive items from sampled triplets '''
        raise NotImplementedError

    def get_neg_items_reprs(self, items):
        ''' how do we represent negative items from sampled triplets '''
        raise NotImplementedError
