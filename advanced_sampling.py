import torch
import random

from base_model import BaseModel
from dataset import BaseDataset


class AdvSamplDataset(BaseDataset):

    def __getitem__(self, idx):
        return idx // self.bucket_len, random.sample(self.range_items, k=min(len(self.range_items), 1000))


class AdvSamplModel(BaseModel):
    '''
        dynamic negative sampling
        ranks 1000 random items, removes positives, and returns top k negatives
    '''
    # TODO: rank all positives and return those which have the lowest scores, instead of taking random positives?

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.positive_lists = dataset.positive_lists

    def score_batchwise(self, users_emb, items_emb, users):
        '''
            calculate scores for all items for users in the batch
            each user has 1000 randomly picked items
            users_emb.shape = (batch_size, emb_size)
            items_emb.shape = (batch_size, 1000, emb_size)
        '''
        return torch.matmul(users_emb.unsqueeze(1), items_emb.transpose(1, 2)).squeeze()

    def get_loss(self, data):
        '''
            rank all items from user-batch, select top-k negative items
            compute loss for them crossprod 5 positive per user
        '''

        users_emb, items_emb = self.representation
        users, items = data
        items = torch.stack(items).t()
        rankings = self.score_batchwise(users_emb[users], items_emb[items], users)
        batch = []

        for user, item, rank in zip(users, items, rankings):
            item = item[rank.argsort(descending=True)]  # resort by score
            positives = self.positive_lists[user]['list']
            positives = torch.tensor(random.sample(positives, min(5, len(positives))))
            negatives = self.subtract_tensor_as_set(item, self.positive_lists[user]['tensor'])[:max(self.k)]
            prod = torch.cartesian_prod(positives, negatives)
            batch.append(torch.cat([user.expand(prod.shape[0], 1), prod], dim=1))

        return super().get_loss(torch.cat(batch))

    def subtract_tensor_as_set(self, t1, t2):
        '''
            quickly subtracts elements of the second tensor from the first tensor as if they were sets
            copied from stackoverflow. no clue how this works
        '''
        return t1[(t2.repeat(t1.shape[0], 1).T != t1).T.prod(1) == 1]
