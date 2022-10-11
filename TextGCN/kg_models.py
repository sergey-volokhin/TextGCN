import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from .dataset import BaseDataset
from .text_base_model import TextBaseModel
from .utils import embed_text


class DatasetKG(BaseDataset):

    def __init__(self, args):
        super().__init__(args)
        self._load_kg(args.bert_model, args.emb_batch_size, args.sep)

    def _load_kg(
        self,
        bert_model: str = 'bert-base-uncased',
        emb_batch_size: int = 64,
        sep: str = '[SEP]'
    ) -> None:

        emb_file = f'{self.path}/embeddings/item_kg_repr_{bert_model.split("/")[-1]}.torch'
        if os.path.exists(emb_file):
            self.items_as_desc = torch.load(emb_file, map_location=self.device)
            return

        self.kg_df_text = pd.read_table(self.path + 'kg_readable.tsv', dtype=str)[['asin', 'relation', 'attribute']]
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
            self.device
        )


class TextModelKG(TextBaseModel):

    def __init__(self, args, dataset) -> None:
        super().__init__(args, dataset)

        ''' all items are represented with their descriptions '''

        if args.pos == 'kg' or args.model == 'kg':
            self.get_pos_items_reprs = self.get_item_desc
        if args.neg == 'kg' or args.model == 'kg':
            self.get_neg_items_reprs = self.get_item_desc

    def _copy_dataset_args(self, dataset) -> None:
        super()._copy_dataset_args(dataset)
        self.items_as_desc = dataset.items_as_desc

    def get_item_desc(self, items, *args):
        return self.items_as_desc[items]
