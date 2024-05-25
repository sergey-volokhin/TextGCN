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
        self.objective = 'ranking'
        super()._load_files(*args, **kwargs)

    def _copy_params(self, config):
        super()._copy_params(config)
        self.k = config.k

    def _build_dicts(self):
        super()._build_dicts()
        # split test into batches once at init instead of at every predict
        # list of lists with test samples (per user), used for evaluation
        self.true_test_lil = self.test_df.groupby('user_id')['asin'].aggregate(list).values.tolist()

    def _train_test_split(self, df, train_size=0.8):
        '''
        split users into two groups: those with 2 reviews and those with more than 2 reviews
        ensure that each user is in both train and test set by selecting at least one review per user
        '''

        vc = df.user_id.value_counts()
        two_reviews = df[df.user_id.isin(vc[vc == 2].keys())]
        more_than_two = df.drop(two_reviews.index)

        train_two = two_reviews.groupby('user_id').sample(n=1, random_state=self.seed)
        test_two = two_reviews[~two_reviews.index.isin(train_two.index)]

        train_more_than_two, test_more_than_two = tts(
            more_than_two,
            train_size=train_size,
            random_state=self.seed,
            stratify=more_than_two.user_id,
        )

        self.train_df = pd.concat([train_two, train_more_than_two])
        self.test_df = pd.concat([test_two, test_more_than_two])
