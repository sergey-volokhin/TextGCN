import torch
import torch.nn.functional as F

from base_model import BaseModel
from utils import sent_trans_embed_text


class TextModelReviewsLoss(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.path = args.data
        self.bert_model = args.bert_model
        self.emb_batch_size = args.emb_batch_size

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.reviews = dataset.reviews
        self.num_reviews = dataset.num_reviews
        self.user_mapping = dataset.user_mapping
        self.item_mapping = dataset.item_mapping
        self.train_user_dict = dataset.train_user_dict

    def _init_embeddings(self, emb_size):
        super()._init_embeddings(emb_size)

        ''' load/calc embeddings of the reviews and setup the dicts '''
        emb_file = f'{self.path}/embeddings/item_reviews_loss_repr_{self.bert_model.split("/")[-1]}.torch'
        self.reviews['vector'] = sent_trans_embed_text(self.reviews['review'],
                                                       emb_file,
                                                       self.bert_model,
                                                       self.emb_batch_size,
                                                       self.device,
                                                       self.logger).cpu().numpy().tolist()

        item_text_embs = {}
        for item, group in self.reviews.groupby('asin')['vector']:
            item_text_embs[item] = torch.tensor(group.values.tolist()).mean(axis=0)
        self.item_text_embs = torch.stack(self.item_mapping['org_id'].map(item_text_embs).values.tolist())

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
        pos_emb = item_emb[pos]
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        loss = 0
        for neg in negs:
            neg_emb = item_emb[neg]
            neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores)
            semantic_regularization = self.semantic_loss(pos, neg, pos_scores, neg_scores)
            loss += torch.mean(bpr_loss + semantic_regularization)
        return loss / len(negs)

    def semantic_loss(self, pos, neg, pos_scores, neg_scores):
        ''' get semantic regularization using textual embeddings '''
        weight = 1
        distance = torch.max(self.bert_sim(pos, neg), torch.tensor(0))
        semantic_regularization = weight * distance
        self.sem_reg += semantic_regularization.mean()
        return semantic_regularization

    def bert_sim(self, pos, neg):
        cands = self.item_mapping_emb[pos.cpu()]
        refs = self.item_mapping_emb[neg.cpu()]
        return self.sim_fn(cands, refs).to(self.device)

    def gnn_sim(self, pos, neg):
        cands = self.embedding_item(pos)
        refs = self.embedding_item(neg)
        return self.sim_fn(cands, refs).to(self.device)
