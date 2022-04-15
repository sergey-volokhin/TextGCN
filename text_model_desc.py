import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from base_model import BaseModel
from dataset import BaseDataset
from utils import sent_trans_embed_text


class DatasetKG(BaseDataset):

    def __init__(self, args):
        super().__init__(args)
        self._construct_text_representation()

    def _copy_args(self, args):
        super()._copy_args(args)
        self.sep = args.sep

    def _load_files(self):
        super()._load_files()
        self.kg_df_text = pd.read_table(self.path + 'kg_readable.tsv', dtype=str)[['asin', 'relation', 'attribute']]

    def _construct_text_representation(self):
        item_text_dict = {}
        for asin, group in tqdm(self.kg_df_text.groupby('asin'),
                                desc='construct text repr',
                                dynamic_ncols=True,
                                disable=self.slurm):
            vals = group[['relation', 'attribute']].values
            item_text_dict[asin] = f' {self.sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping['text'] = self.item_mapping['org_id'].map(item_text_dict)


class TextModelKG(BaseModel):

    def _copy_args(self, args):
        super()._copy_args(args)
        self.bert_model = args.bert_model
        self.emb_batch_size = args.emb_batch_size

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.path = dataset.path
        self.item_mapping = dataset.item_mapping
        self.train_user_dict = dataset.train_user_dict

    def _init_embeddings(self, emb_size):
        ''' construct BERT representation for items and overwrite model item embeddings '''
        super()._init_embeddings(emb_size)

        emb_file = f'{self.path}/embeddings/item_kg_repr_{self.bert_model.split("/")[-1]}.torch'
        self.stats_dict = sent_trans_embed_text(self.item_mapping['text'].tolist(),
                                                emb_file,
                                                self.bert_model,
                                                self.emb_batch_size,
                                                self.device,
                                                self.logger)

    @property
    def representation(self):
        curent_lvl_emb_matrix = self.embedding_matrix
        if self.training:
            norm_matrix = self._dropout_norm_matrix
        else:
            norm_matrix = self.norm_matrix
        node_embed_cache = [curent_lvl_emb_matrix]
        for _ in range(self.n_layers):
            curent_lvl_emb_matrix = self.layer_propagate(norm_matrix, curent_lvl_emb_matrix)
            node_embed_cache.append(curent_lvl_emb_matrix)
        aggregated_embeddings = self.layer_aggregation(node_embed_cache)
        return torch.split(aggregated_embeddings, [self.n_users, self.n_items])

    def layer_aggregation(self, vectors):
        return torch.mean(torch.stack(vectors, dim=1), dim=1)

    def layer_propagate(self, norm_matrix, emb_matrix):
        return torch.sparse.mm(norm_matrix, emb_matrix)

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
            semantic_regularization = self.semantic_reg_los(pos, neg, pos_scores, neg_scores)
            loss += torch.mean(bpr_loss + semantic_regularization)
        return loss / len(negs)

    def semantic_reg_los(self, pos, neg, pos_scores, neg_scores):
        ''' get semantic regularization using textual embeddings '''

        weight = 1
        # weight = max(neg_scores - pos_scores, 0) <- useless since pos will be higher than neg in long term
        # weight = max(pos_scores - neg_scores, 0)
        # weight = torch.abs(pos_scores - neg_scores)

        # distance = torch.abs(self.bert_sim(pos, neg) - self.gnn_sim(pos, neg))
        # distance = torch.max(self.bert_sim(pos, neg) - self.gnn_sim(pos, neg), torch.tensor(0))
        # distance = torch.max(self.gnn_sim(pos, neg) - self.bert_sim(pos, neg), torch.tensor(0))
        distance = torch.max(self.bert_sim(pos, neg), torch.tensor(0))
        # distance = torch.abs(self.bert_sim(pos, neg) * self.gnn_sim(pos, neg))

        semantic_regularization = weight * distance
        self.sem_reg += semantic_regularization.mean()
        return semantic_regularization

    def bert_sim(self, pos, neg):
        cands = self.item_mapping['text'].values[pos.cpu()].tolist()
        refs = self.item_mapping['text'].values[neg.cpu()].tolist()
        return F.cosine_similarity(torch.stack([self.stats_dict[i] for i in cands]),
                                   torch.stack([self.stats_dict[i] for i in refs]))

    def gnn_sim(self, pos, neg):
        return F.cosine_similarity(self.embedding_item(pos), self.embedding_item(neg))
