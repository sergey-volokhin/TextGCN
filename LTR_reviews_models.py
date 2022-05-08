import torch

from reviews_models import DatasetReviews, TextModelReviews
from kg_models import DatasetKG


class LTRDataset(DatasetKG, DatasetReviews):

    def __init__(self, args):
        super().__init__(args)
        self._get_users_as_avg_reviews()

    def _get_users_as_avg_reviews(self):
        ''' use average of reviews to represent users '''
        user_text_embs = {}
        for user, group in self.top_med_reviews.groupby('user_id')['vector']:
            user_text_embs[user] = torch.tensor(group.values.tolist()).mean(axis=0)
        self.users_as_avg_reviews = torch.stack(self.user_mapping['remap_id'].map(
            user_text_embs).values.tolist()).to(self.device)


class LTR(TextModelReviews):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def score(self, users, items, users_emb, item_emb):
        user_part = torch.cat([users_emb, self.users_as_avg_reviews[users]], axis=1)
        item_part = torch.cat([item_emb[items], self.items_as_avg_reviews[items]], axis=1)
        # item_part = torch.cat([item_emb[items], self.items_as_desc[items]], axis=1)
        return torch.sum(torch.mul(user_part, item_part), dim=1)
        # return torch.sum(torch.mul(users_emb, item_emb[items]), dim=1)
