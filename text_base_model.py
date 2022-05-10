import torch
import torch.nn.functional as F

from base_model import BaseModel


class TextBaseModel(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.dist_fn = args.dist_fn
        self.weight = args.weight

    def _add_vars(self):
        super()._add_vars()
        self.dist_fn = {
            'euclid': F.pairwise_distance,
            'cosine_minus': lambda x, y: -F.cosine_similarity(x, y),
            'cosine_ratio': lambda x, y: torch.nan_to_num(1 / F.cosine_similarity(x, y),
                                                          posinf=0,
                                                          neginf=0),
        }[self.dist_fn]

    def bpr_loss(self, users, pos, negs):
        users_emb, item_emb = self.representation
        users_emb = users_emb[users]
        pos_scores = self.score(users, pos, users_emb, item_emb[pos])
        loss = 0
        for neg in negs:
            neg_scores = self.score(users, neg, users_emb, item_emb[neg])
            bpr_loss = F.softplus(neg_scores - pos_scores)
            sem_loss = self.semantic_loss(users, pos, neg, pos_scores, neg_scores)
            self._bpr_loss += torch.mean(bpr_loss) / len(negs)
            loss += torch.mean(bpr_loss + sem_loss)
        return loss / len(negs)

    def semantic_loss(self, users, pos, neg, pos_scores, neg_scores):
        ''' get semantic regularization using textual embeddings '''

        b = self.bert_dist(users, pos, neg)
        g = self.gnn_dist(pos, neg)
        distance = {'b': b,
                    'max(b)': F.relu(b),
                    'max(b-g)': F.relu(b - g),
                    'max(g-b)': F.relu(g - b),
                    '(b-g)': b - g,
                    '(g-b)': g - b,
                    '|b-g|': torch.abs(b - g),
                    '|g-b|': torch.abs(g - b),
                    }[self.weight.split('_')[1]]

        weight = {'max(p-n)': F.relu(pos_scores - neg_scores),
                  '|p-n|': torch.abs(pos_scores - neg_scores),
                  '1': 1,
                  '0': 0,  # in case we want to not have semantic loss
                  }[self.weight.split('_')[0]]

        semantic_loss = weight * distance
        self._sem_loss += semantic_loss.mean()
        return semantic_loss

    def gnn_dist(self, pos, neg):
        ''' calculate similarity between gnn representations of the sampled items '''
        cands = F.normalize(self.embedding_item(pos))
        refs = F.normalize(self.embedding_item(neg))
        return self.dist_fn(cands, refs).to(self.device)

    def bert_dist(self, users, pos, neg):
        ''' calculate similarity between textual representations of the sampled items '''
        cands = self.pos_item_reprs(users, pos)
        refs = self.neg_item_reprs(users, neg)
        return self.dist_fn(cands, refs).to(self.device)

    def pos_item_reprs(self, users, items):
        ''' how do we represent positive items from sampled triplets '''
        raise NotImplementedError

    def neg_item_reprs(self, users, items):
        ''' how do we represent negative items from sampled triplets '''
        raise NotImplementedError
