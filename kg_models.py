import pandas as pd
from tqdm import tqdm

from dataset import BaseDataset
from text_model import TextBaseModel
from utils import embed_text


class DatasetKG(BaseDataset):

    def __init__(self, args):
        super().__init__(args)
        self._load_kg(args.sep, args.emb_batch_size, args.bert_model)

    def _load_kg(self, sep, emb_batch_size, bert_model):
        self.kg_df_text = pd.read_table(self.path + 'kg_readable.tsv', dtype=str)[['asin', 'relation', 'attribute']]
        item_text_dict = {}
        for asin, group in tqdm(self.kg_df_text.groupby('asin'),
                                desc='kg text repr',
                                dynamic_ncols=True,
                                disable=self.slurm):
            vals = group[['relation', 'attribute']].values
            item_text_dict[asin] = f' {sep} '.join([f'{relation}: {attribute}' for (relation, attribute) in vals])
        self.item_mapping['text'] = self.item_mapping['org_id'].map(item_text_dict)

        emb_file = f'{self.path}/embeddings/item_kg_repr_{bert_model.split("/")[-1]}.torch'
        self.items_as_desc = embed_text(self.item_mapping['text'],
                                        emb_file,
                                        bert_model,
                                        emb_batch_size,
                                        self.device,
                                        self.logger)


class TextModelKG(TextBaseModel):

    def _copy_dataset_args(self, dataset):
        super()._copy_dataset_args(dataset)
        self.items_as_desc = dataset.items_as_desc

    def bert_sim(self, users, pos, neg):
        cands = self.items_as_desc[pos.cpu()]
        refs = self.items_as_desc[neg.cpu()]
        return self.sim_fn(cands, refs).to(self.device)
