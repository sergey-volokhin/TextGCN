import pandas as pd
from sklearn.model_selection import train_test_split as tts

from .BaseDataset import BaseDataset


class DatasetRanking(BaseDataset):
    '''
    dataset for ranking task:
        1. does not have val set
        2. pre-split test set into batches for speed
    '''

    def _safety_checks(self):
        super()._safety_checks()
        assert self.n_items > max(self.k), f'all k must be less than number of items ({self.n_items}), got k={self.k}'

    def _load_files(self, *args, **kwargs):
        super()._load_files(*args, **kwargs)
        if hasattr(self, 'val_df'):
            self.test_df = pd.concat([self.val_df, self.test_df])
            delattr(self, 'val_df')

    def _copy_params(self, config):
        super()._copy_params(config)
        self.k = config.k

    def _build_dicts(self):
        super()._build_dicts()
        # split test into batches once at init instead of at every predict
        # list of lists with test samples (per user), used for evaluation
        self.true_test_lil = self.test_df.groupby('user_id')['asin'].aggregate(list).values.tolist()

    def _train_test_split(self, df, train_size=0.8):
        self.train_df, self.test_df = tts(df, train_size=train_size, stratify=df['user_id'], random_state=self.seed)
