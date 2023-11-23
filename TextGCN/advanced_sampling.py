import random

import torch

from .base_model import BaseModel
from .dataset import BaseDataset
from .utils import subtract_tensor_as_set


class AdvSamplDataset(BaseDataset):

    pos_samples = 5
    max_neg_samples = 1000

    def __init__(self, params):
        super().__init__(params)
        # is creating and referencing range faster than recreating range every time when sampling?
        self.range_items = range(self.n_items)
        self.neg_samples = min(len(self.range_items), self.max_neg_samples)

    def __getitem__(self, idx):
        return torch.tensor([idx // self.bucket_len] + random.sample(self.range_items, k=self.neg_samples))


class AdvSamplModel(BaseModel):
    '''
        dynamic negative sampling
        ranks 1000 random items, removes positives, and returns top k negatives
    '''
    # TODO: rank all positives and return those which have the lowest scores, instead of taking random positives?

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.positive_lists = dataset.positive_lists
        self.pos_samples = dataset.pos_samples

    def score_pairwise_adv(self, users_emb, items_emb):
        '''
            each user has a batch with corresponding items,
            calculate scores for all items for those items:
            users_emb.shape = (batch_size, emb_size)
            items_emb.shape = (batch_size, 1000, emb_size)
        '''
        return torch.matmul(users_emb.unsqueeze(1), items_emb.transpose(1, 2)).squeeze()

    def get_loss(self, data):
        '''
            rank items per user, select top-k negative items,
            query super() loss for all pairings between those negatives
            with self.pos_samples random positives per user

            num pairs in final batch: 5 * k
        '''

        users_emb, items_emb = self.representation
        data = data.to(self.device)
        users, items = data[:, 0], data[:, 1:]
        rankings = self.score_pairwise_adv(users_emb[users], items_emb[items])
        batch = []

        for user, item, rank in zip(users, items, rankings):
            item = item[rank.argsort(descending=True)]  # sort items by score to select highest rated
            positives = self.positive_lists[user]['list']
            positives = torch.tensor(random.sample(positives, min(self.pos_samples, len(positives)))).to(self.device)
            negatives = subtract_tensor_as_set(item, self.positive_lists[user]['tensor'])[:max(self.k)]
            prod = torch.cartesian_prod(positives, negatives)
            batch.append(torch.cat([user.expand(prod.shape[0], 1), prod], dim=1))

        return super().get_loss(torch.cat(batch))
