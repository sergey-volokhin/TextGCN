import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from kg_models import DatasetKG
from reviews_models import DatasetReviews, TextModelReviews
from base_model import BaseModel


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

    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        if args.model == 'ltr_reviews':
            self.score = self.score_reviews
        elif args.model == 'ltr_kg':
            self.score = self.score_kg
        else:
            raise AttributeError(f'wrong LTR model name: "{args.model}"')

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def score_reviews(self, users, items, users_emb, item_emb):
        user_part = torch.cat([users_emb, self.users_as_avg_reviews[users]], axis=1)
        item_part = torch.cat([item_emb[items], self.items_as_avg_reviews[items]], axis=1)
        return torch.sum(torch.mul(user_part, item_part), dim=1)

    def score_kg(self, users, items, users_emb, item_emb):
        user_part = torch.cat([users_emb, self.users_as_avg_reviews[users]], axis=1)
        item_part = torch.cat([item_emb[items], self.items_as_desc[items]], axis=1)
        return torch.sum(torch.mul(user_part, item_part), dim=1)


class LTRSimple(BaseModel):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.users_as_avg_reviews = dataset.users_as_avg_reviews
        self.items_as_avg_reviews = dataset.items_as_avg_reviews
        self.items_as_desc = dataset.items_as_desc

    def predict(self, save=False):
        '''
            using new formula:
            :math:`\sigma((\mathbf{e}_{u,gnn}||\mathbf{e}_{u,text})\cdot(\mathbf{e}_{neg,gnn}||\mathbf{e}_{neg,text}) - \\
            (\mathbf{e}_{u,gnn}||\mathbf{e}_{u,text})\cdot(\mathbf{e}_{pos, gnn}||\mathbf{e}_{pos,text}))`
        '''
        users = list(self.test_user_dict)
        y_pred, y_true = [], []
        with torch.no_grad():  # don't calculate gradient since we only predict
            users_emb, items_emb = self.representation
            full_items_emb = torch.cat([items_emb, self.items_as_avg_reviews], axis=1)

            batches = [users[j:j + self.batch_size] for j in range(0, len(users), self.batch_size)]
            batches = tqdm(batches, desc='batches', leave=False, dynamic_ncols=True, disable=self.slurm)
            for batch_users in batches:

                # get the estimated user-item scores with matmul embedding matrices
                batch_user_emb = users_emb[torch.tensor(batch_users).long().to(self.device)]
                full_batch_user_emb = torch.cat([batch_user_emb, self.users_as_avg_reviews[batch_users]], axis=1)
                rating = torch.sigmoid(torch.matmul(full_batch_user_emb, full_items_emb.t()))

                # set scores for train items to be -inf so we don't recommend them
                exclude_index, exclude_items = [], []
                for ind, items in enumerate(self.get_user_pos_items(batch_users)):
                    exclude_index += [ind] * len(items)
                    exclude_items.append(items)
                exclude_items = np.concatenate(exclude_items)
                rating[exclude_index, exclude_items] = np.NINF

                # select top-k items with highest ratings
                _, rank_indices = torch.topk(rating, k=max(self.k))
                y_pred += list(rank_indices.cpu().numpy().tolist())
                y_true += [self.test_user_dict[u] for u in batch_users]

        predictions = pd.DataFrame.from_dict({'y_pred': y_pred, 'y_true': y_true})
        if save:
            predictions.to_csv(f'{self.save_path}/predictions.tsv', sep='\t', index=False)
            self.logger.info(f'Predictions are saved in `{self.save_path}/predictions.tsv`')
        return predictions
