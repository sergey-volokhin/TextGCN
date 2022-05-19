import torch
import random

from base_model import BaseModel
from dataset import BaseDataset


class AdvSamplDataset(BaseDataset):

    def __getitem__(self, idx):
        return idx // self.bucket_len, random.sample(self.range_items, k=1000)


class AdvSamplModel(BaseModel):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.positive_lists = dataset.positive_lists

    def old_loss(self, *args, **kwargs):
        return super().get_loss(*args, **kwargs)

    def rank_items_for_batch_pairwise(self, users_emb, items_emb, batch_users):
        ''' calculate scores for all items for users in the batch '''
        return torch.matmul(users_emb.unsqueeze(1), items_emb.transpose(1, 2)).squeeze()

    def get_loss(self, data):
        '''
            rank all items from user-batch, select negative items from top k
            compute loss for them crossprod 5 positive per user
        '''
        users_emb, items_emb = self.representation
        users, items = data
        items = torch.stack(items).t()
        rankings = self.rank_items_for_batch_pairwise(users_emb[users], items_emb[items], users)
        batch = []
        for user, item, rank in zip(users, items, rankings):

            top_items = rank.argsort(descending=True)
            item = item[top_items]

            positives = self.positive_lists[user]['list']
            positives = torch.tensor(random.sample(positives, min(5, len(positives))))
            negatives = self.subtract_tensor_as_set(item, self.positive_lists[user]['tensor'])[:max(self.k)]

            prod = torch.cartesian_prod(positives, negatives)
            batch.append(torch.cat([user.expand(prod.shape[0], 1), prod], dim=1))

        new_data = torch.cat(batch)
        return self.old_loss(new_data)

    def subtract_tensor_as_set(self, t1, t2):
        return t1[(t2.repeat(t1.shape[0], 1).T != t1).T.prod(1) == 1]
