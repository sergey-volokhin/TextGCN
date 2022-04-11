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
        for asin, group in tqdm(self.kg_df_text.groupby('asin'), desc='construct text repr', dynamic_ncols=True):
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

    @property
    def embedding_matrix(self):
        ''' get the embedding matrix of 0th layer '''
        return torch.cat([self.embedding_user.weight, self.embedding_item.weight])

    def get_loss(self, users, pos, neg):

        ''' normal loss '''
        # TODO change to only return representation of (users, pos, neg)
        users_emb, item_emb = self.representation
        users_emb = users_emb[users]
        pos_emb = item_emb[pos]
        neg_emb = item_emb[neg]
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        old_loss = F.softplus(neg_scores - pos_scores)

        weight = neg_scores - pos_scores
        new_loss = torch.abs(weight * (self.bert_sim(pos, neg) - self.gnn_sim(pos, neg)))
        loss = torch.mean(old_loss + new_loss)

        ''' regularization loss '''
        user_vec = self.embedding_user(users)
        pos_vec = self.embedding_item(pos)
        neg_vec = self.embedding_item(neg)
        reg_loss = (user_vec.norm(2).pow(2) +
                    pos_vec.norm(2).pow(2) +
                    neg_vec.norm(2).pow(2)) / len(users) / 2

        return loss + self.reg_lambda * reg_loss

    def bert_sim(self, pos, neg):
        cands = self.item_mapping['text'].values[pos.cpu()].tolist()
        refs = self.item_mapping['text'].values[neg.cpu()].tolist()
        cands_norm = F.normalize(torch.stack([self.stats_dict[i] for i in cands]), p=2, dim=1)
        refs_norm = F.normalize(torch.stack([self.stats_dict[i] for i in refs]), p=2, dim=1)
        return (cands_norm * refs_norm).sum(axis=1).to(self.device)

    def gnn_sim(self, pos, neg):
        pos_norm = F.normalize(self.embedding_item(pos), p=2, dim=1)
        neg_norm = F.normalize(self.embedding_item(neg), p=2, dim=1)
        return (pos_norm * neg_norm).sum(axis=1).to(self.device)
