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

    def __init__(self, config):
        super().__init__(config)
        self._load_kg(encoder=config.encoder, emb_batch_size=config.emb_batch_size, sep=config.sep)

    def _load_kg(
        self,
        encoder: str,
        emb_batch_size: int = 64,
        sep: str = '[SEP]',
    ):

        emb_file = os.path.join(
            self.path,
            'embeddings',
            f'item_kg_repr_{encoder.split("/")[-1]}_{self.seed}-seed.torch',
        )
        if os.path.exists(emb_file):
            self.items_as_desc = torch.load(emb_file, map_location=self.device)
            return

        kg_df_text = pd.read_table(os.path.join(self.path, 'kg_readable.tsv'),
                                   usecols=['asin', 'relation', 'attribute'],
                                   dtype=str)
        item_text_dict = {}
        for asin, group in tqdm(kg_df_text.groupby('asin')[['relation', 'attribute']],
                                desc='kg text repr',
                                dynamic_ncols=True,
                                leave=False,
                                disable=self.slurm):
            item_text_dict[asin] = f' {sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in group.values])

        self.item_mapping['text'] = self.item_mapping['org_id'].map(item_text_dict)
        self.items_as_desc = embed_text(
            self.item_mapping['text'],
            emb_file,
            encoder,
            emb_batch_size,
            self.device,
        )
