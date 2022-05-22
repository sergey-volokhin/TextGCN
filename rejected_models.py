import torch

from base_model import BaseModel


class LTRCosine(BaseModel):
    ''' train the LightGCN model from scratch, concatenating GNN vectors with text '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.users_text_repr = self.users_as_avg_reviews
        self.items_text_repr = {'ltr_reviews': self.items_as_avg_reviews,
                                'ltr_kg': self.items_as_desc,
                                }[args.model]

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
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
        concatenates textual repr to LightGCN vectors during inference
    '''

    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.users_text_repr = self.users_as_avg_reviews

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.reviews_df = dataset.reviews_df
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
