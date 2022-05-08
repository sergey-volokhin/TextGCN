import torch
import torch.nn.functional as F

from base_model import BaseModel


class TextBaseModel(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.sim_fn = args.sim_fn
        self.weight = args.weight

    def _add_vars(self):
        super()._add_vars()
        self.sim_fn = {
            'cosine': F.cosine_similarity,
            'euclid_minus': lambda x, y: -F.pairwise_distance(x, y),
            'euclid_ratio': lambda x, y: torch.nan_to_num(1 / F.pairwise_distance(x, y),
                                                          posinf=0,
                                                          neginf=0),
        }[self.sim_fn]

    def bpr_loss(self, users, pos, negs):
        users_emb, item_emb = self.representation
        users_emb = users_emb[users]
        pos_scores = self.score(users, pos, users_emb, item_emb)
        loss = 0
        for neg in negs:
            neg_scores = self.score(users, neg, users_emb, item_emb)
            bpr_loss = F.softplus(neg_scores - pos_scores)
            semantic_regularization = self.semantic_loss(users, pos, neg, pos_scores, neg_scores)
            loss += torch.mean(bpr_loss + semantic_regularization)
        return loss / len(negs)

    def semantic_loss(self, users, pos, neg, pos_scores, neg_scores):
        ''' get semantic regularization using textual embeddings '''

        if 'max(p-n)' in self.weight:
            weight = F.relu(pos_scores - neg_scores)
        elif '|p-n|' in self.weight:
            weight = torch.abs(pos_scores - neg_scores)
        elif '_1_' in self.weight:
            weight = 1
        else:
            raise AttributeError('incorrect weight')

        b = self.bert_sim(users, pos, neg)
        g = self.gnn_sim(pos, neg)
        if 'max(b)' in self.weight:
            distance = F.relu(b)
        elif 'max(b-g)' in self.weight:
            distance = F.relu(b - g)
        elif '(b-g)' in self.weight:
            distance = b - g
        elif 'max(g-b)' in self.weight:
            distance = F.relu(g - b)
        elif '(g-b)' in self.weight:
            distance = g - b
        elif '|b-g|' in self.weight:
            distance = torch.abs(b - g)
        elif '|g-b|' in self.weight:
            distance = torch.abs(g - b)
        elif '_b' in self.weight:
            distance = b
        else:
            raise AttributeError('incorrect weights')

        semantic_regularization = weight * distance
        self.sem_reg += semantic_regularization.mean()
        return semantic_regularization

    def gnn_sim(self, pos, neg):
        cands = self.embedding_item(pos)
        refs = self.embedding_item(neg)
        return self.sim_fn(cands, refs).to(self.device)

    def bert_sim(self, *args):
        raise NotImplementedError
