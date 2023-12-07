import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from .BaseDataset import BaseDataset
from .utils import embed_text


class DatasetKG(BaseDataset):
    '''
    Dataset with items' features from knowledge graph
    calculates textual representations of items and users using item's descriptions
    '''

    def __init__(self, params):
        super().__init__(params)
        self._load_kg(params.bert_model, params.emb_batch_size, params.sep)
        self._get_users_as_avg_desc()

    def _load_kg(
        self,
        bert_model: str = 'bert-base-uncased',
        emb_batch_size: int = 64,
        sep: str = '[SEP]',
    ):

        emb_file = os.path.join(
            self.path,
            'embeddings',
            f'item_kg_repr_{bert_model.split("/")[-1]}_{self.seed}-seed.torch',
        )
        if os.path.exists(emb_file):
            self.items_as_desc = torch.load(emb_file, map_location=self.device)
            return

        self.kg_df_text = pd.read_table(os.path.join(self.path, 'kg_readable.tsv'),
                                        usecols=['asin', 'relation', 'attribute'],
                                        dtype=str)
        item_text_dict = {}
        for asin, group in tqdm(self.kg_df_text.groupby('asin'),
                                desc='kg text repr',
                                dynamic_ncols=True,
                                leave=False,
                                disable=self.slurm):
            vals = group[['relation', 'attribute']].values
            item_text_dict[asin] = f' {sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])

        self.item_mapping['text'] = self.item_mapping['org_id'].map(item_text_dict)
        self.items_as_desc = embed_text(
            self.item_mapping['text'],
            emb_file,
            bert_model,
            emb_batch_size,
            self.device,
        )

    def _get_users_as_avg_desc(self):
        ''' use mean of descriptions to represent users '''
        user_text_embs = {}
        for user, group in self.top_med_reviews.groupby('user_id')['asin']:
            user_text_embs[user] = self.items_as_desc[group.values].mean(axis=0).cpu()
        self.users_as_avg_desc = torch.stack(
            self.user_mapping['remap_id']
            .map(user_text_embs)
            .values
            .tolist()
        ).to(self.device)
