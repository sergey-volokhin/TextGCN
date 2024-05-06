import os

import pandas as pd

from .BaseDataset import BaseDataset
from .utils import embed_text


class DatasetProfile(BaseDataset):
    '''
    Dataset using generated profiles to represent users
    calculates textual representations of items and users using user profiles
    '''

    def __init__(self, config):
        super().__init__(config)
        self._load_profiles(model_name=config.encoder, emb_batch_size=config.emb_batch_size)

    def _load_profiles(
        self,
        model_name: str,
        emb_batch_size: int = 64,
    ):
        '''
        load profiles, embed them and save embeddings as a dict
        '''
        self.logger.debug('loading profiles and getting embeddings')
        emb_file = os.path.join(
            self.path,
            'embeddings',
            f'user_profile_repr_{model_name.split("/")[-1]}_seed{self.seed}.pkl',
        )

        profiles_df = pd.read_table(os.path.join(self.path, f'reshuffle_{self.seed}', 'profiles_0.4_llama2.tsv'), index_col=0)
        profiles_df = profiles_df[profiles_df.index.isin(self.user_mapping.org_id)]
        assert not profiles_df.isna().any().any(), f'missing values in profiles: {profiles_df.isna().any()}'

        self.user_representations['profile'] = embed_text(
            sentences=profiles_df.profile.values.flatten().tolist(),
            path=emb_file,
            model_name=model_name,
            batch_size=emb_batch_size,
            logger=self.logger,
            device=self.device,
        )
