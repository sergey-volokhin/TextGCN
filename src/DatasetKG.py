import os

import pandas as pd

from .BaseDataset import BaseDataset
from .utils import embed_text


class DatasetKG(BaseDataset):
    '''
    Dataset with items' features from knowledge graph
    calculates textual representations of items and users using item's descriptions
    '''

    def __init__(self, config):
        super().__init__(config)
        self._load_kg(model_name=config.encoder, emb_batch_size=config.emb_batch_size)

    def _copy_params(self, config):
        super()._copy_params(config)
        self.kg_features = config.kg_features

    def _load_kg(
        self,
        model_name: str,
        emb_batch_size: int = 64,
    ):
        '''
        load knowledge graph and calculate textual representations of items
        we represent items by descriptions and generated texts (have to start with "generated_")
        embed them and save embeddings as a dict
        '''
        self.logger.debug('loading KG and getting embeddings')
        emb_file = os.path.join(
            self.path,
            'embeddings',
            f'item_kg_repr_{model_name.split("/")[-1]}.pkl',
        )

        kg = pd.read_table(
            os.path.join(self.path, 'kg_readable_w_gen_desc_v1.tsv'),
            usecols=['asin', 'relation', 'attribute'],
            dtype=str,
        ).pivot(index='asin', columns='relation', values='attribute')

        # remove items that don't appear in the training
        kg = kg[kg.index.isin(self.item_mapping.org_id)]

        # create "base description" column from title and seller-provided description if it exists
        kg['base_desc'] = 'Title: "' + kg['title'] + ('"\nDescription: "' + kg['description'] + '"').fillna('')
        # remove columns that won't be embedded
        kg_to_encode = kg[['base_desc'] + [i for i in kg.columns if i.startswith('gen_')]]
        kg_to_encode.columns = ['base_desc'] + [i.replace('gen_', '') for i in kg_to_encode.columns[1:]]

        assert not kg_to_encode.isna().any().any(), f'missing values in kg_to_encode: {kg_to_encode.isna().any()}'

        embeddings = embed_text(
            sentences=kg_to_encode.values.flatten().tolist(),
            path=emb_file,
            model_name=model_name,
            batch_size=emb_batch_size,
            logger=self.logger,
            device=self.device,
        )
        reshaped = embeddings.reshape(*kg_to_encode.shape, embeddings.shape[1])

        # this is a bit fucky
        for c in self.kg_features:
            self.item_representations[c] = reshaped[:, kg_to_encode.columns.tolist().index(c)]
