import os

import pandas as pd
import torch

from .BaseDataset import BaseDataset
from .utils import embed_text


class DatasetProfile(BaseDataset):
    '''
    Dataset using generated profiles to represent users
    calculates textual representations of items and users using user profiles
    '''

    def _load_profiles(self, config):
        '''
        load profiles, embed them and save embeddings as a dict
        represent users by their profiles
        '''
        self.logger.debug('loading profiles and getting embeddings')
        emb_file = os.path.join(
            self.path,
            'embeddings',
            f'user_profile_repr_{config.encoder.split("/")[-1]}_seed{self.seed}.pkl',
        )

        profiles_df = pd.read_table(
            os.path.join(
                self.path,
                f'reshuffle_{self.seed}',
                f'profiles_0.4_{config.profile_generator}.tsv',
            ),
            index_col=0,
        )
        profiles_df = profiles_df[profiles_df.index.isin(self.user_mapping.org_id)]
        assert not profiles_df.isna().any().any(), f'missing values in profiles: {profiles_df.isna().any()}'

        self.user_representations['profiles'] = embed_text(
            sentences=profiles_df.profile.values.flatten().tolist(),
            path=emb_file,
            model_name=config.encoder,
            batch_size=config.emb_batch_size,
            logger=self.logger,
            device=self.device,
        )

        if 'profiles' in self.features['item']['nonkg']:
            self._get_items_as_avg_user_profile()

    def _get_items_as_avg_user_profile(self):
        ''' represent items by the mean of user profiles that reviewed that item '''

        item_profiles = {}
        for item, group in self.top_med_interactions.groupby('asin')['user_id']:
            item_profiles[item] = self.user_representations['profiles'][group.values].mean(axis=0).cpu()

        mapped = self.item_mapping['remap_id'].map(item_profiles).values.tolist()
        self.item_representations['profiles'] = torch.stack(mapped).to(self.device)
