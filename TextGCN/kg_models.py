import os

import pandas as pd
import torch

from .dataset import BaseDataset
from .text_base_model import TextBaseModel
from .utils import embed_text


class DatasetKG(BaseDataset):

    def __init__(self, params):
        super().__init__(params)
        self._load_kg(params.bert_model, params.emb_batch_size, params.sep)

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

        kg = pd.read_table(os.path.join(self.path, 'meta_synced.tsv')).set_index('asin')

        # concatenate all features from KG into one column "text"
        columns = kg.columns
        kg['text'] = ''
        for column in columns[:-1]:
            kg['text'] = kg['text'] + kg[column] + f' {sep} '
        kg['text'] = kg['text'] + kg[columns[-1]]
        item_text_dict = kg['text'].to_dict()

        self.item_mapping['text'] = self.item_mapping['org_id'].map(item_text_dict)
        self.items_as_desc = embed_text(
            self.item_mapping['text'],
            emb_file,
            bert_model,
            emb_batch_size,
            self.device,
        )


class TextModelKG(TextBaseModel):

    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        ''' all items are represented with their descriptions '''

        if params.pos == 'kg' or params.model == 'kg':
            self.get_pos_items_reprs = self.get_item_desc
        if params.neg == 'kg' or params.model == 'kg':
            self.get_neg_items_reprs = self.get_item_desc

    def _copy_dataset_params(self, dataset):
        super()._copy_dataset_params(dataset)
        self.items_as_desc = dataset.items_as_desc

    def get_item_desc(self, items):
        return self.items_as_desc[items]
