import html
import re
import string
import sys
import time
from functools import wraps
from unicodedata import normalize

import numpy as np
import orjson as json
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from tqdm import tqdm
from unidecode import unidecode

printable = string.punctuation + string.ascii_letters + string.digits + ' '
unprintable_pattern = re.compile(f'[^{re.escape(printable)}]')

# default NA values from pd.read_csv
na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
             '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
             'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']

emoji_pattern = re.compile(
    u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0"
    u"\U00002702-\U000027B0\U000024C2-\U0001F251\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+",
    flags=re.UNICODE,
)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f'{func.__name__} took {time.perf_counter() - start:.2f} seconds')
        return res
    return wrapper


def clean_text_series(series):
    def remove_html_tags(text):
        return re.sub(r'<[^<]+?>', '', text)

    def remove_emojis(text):
        return emoji_pattern.sub(r'', text)

    series = (
        series.fillna('')
        .apply(unidecode)
        .apply(html.unescape)
        .apply(lambda x: normalize('NFKD', x))
        .apply(remove_html_tags)
        .apply(remove_emojis)
        .str.replace(unprintable_pattern, '', regex=True)
        .str.replace('[\s_]+', ' ', regex=True)
        .str.lower()
        .str.strip(string.punctuation)
        .replace(na_values, np.nan)
    )
    return series[series.str.len() > 1]


@timeit
def process_metadata(path):
    '''
    remove all unused fields,
    normalize textual fields

    returns: pd.DataFrame(   # easier to extend to new elements in KG
        [
            ('asin': asin, 'title': title),
            ('asin': asin, 'description': description)
        ]
    )
    '''
    metadata = open(path, 'r').read().split('\n')
    fields = ['title', 'description', 'asin']
    cleaned = []
    for row in tqdm(metadata, desc='processing metadata', dynamic_ncols=True, leave=False):
        if not row:
            continue
        row = json.loads(row)
        if all(i in row for i in fields):
            cleaned.append({k: row[k] for k in fields})
    df = pd.DataFrame(cleaned).drop_duplicates('asin')
    df['description'] = clean_text_series(df['description'].apply(' '.join))
    return df.replace(na_values, np.nan).dropna().reset_index(drop=True)


@timeit
def process_reviews(path):
    '''
    remove all unused fields from reviews,
    normalize textual fields
    '''
    reviews = open(path, 'r').read().split('\n')
    fields = ['reviewText', 'reviewerID', 'asin', 'unixReviewTime', 'overall']
    cleaned = []
    for row in tqdm(reviews, desc='proc reviews', dynamic_ncols=True, leave=False):
        if not row:
            continue
        row = json.loads(row)
        if all(i in row for i in fields):
            cleaned.append({k: row[k] for k in fields})
    df = (
        pd.DataFrame(cleaned)
        .rename(columns={'reviewerID': 'user_id', 'reviewText': 'review', 'unixReviewTime': 'time'})
        .drop_duplicates(subset=['user_id', 'asin'])
        .replace(na_values, np.nan)
        .dropna()
        .astype({'overall': int})
    )
    df.review = clean_text_series(df.review)
    return df.dropna().reset_index(drop=True)


def intersect(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column: str = 'asin',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ''' remove all items that don't appear in both dataframes '''
    intersection = set(df1[column].unique()).intersection(set(df2[column].unique()))
    df1 = df1[df1[column].isin(intersection)]
    df2 = df2[df2[column].isin(intersection)]
    return df1, df2


def core_n(
    reviews: pd.DataFrame,
    n: int = 5,
    columns: tuple[str, str] = ('asin', 'user_id'),
) -> pd.DataFrame:
    ''' repeatedly
    remove all items that have less than n reviews,
    remove all users that have less than n reviews
    '''
    while True:
        shape = reviews.shape
        for c in columns:
            vc = reviews[c].value_counts()
            reviews = reviews[reviews[c].isin(vc[vc >= n].index)]
        if reviews.shape == shape:
            return reviews


@timeit
def sync(
    meta: pd.DataFrame,
    reviews: pd.DataFrame,
    n: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n == 1:
        return intersect(meta, reviews)
    while True:
        r_shape = reviews.shape
        m_shape = meta.shape
        reviews = core_n(reviews, n)
        meta, reviews = intersect(meta, reviews)
        if reviews.shape == r_shape and meta.shape == m_shape:
            return meta, reviews


@timeit
def train_test_split(
    df: pd.DataFrame,
    column: str = 'user_id',
    train_size: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ''' split df into train and test, ensuring that all users are in both sets '''

    # Pre-filter groups with less than 3 elements
    group_sizes = df.groupby(column).size()
    valid_groups = group_sizes[group_sizes >= 3].index
    filtered_df = df[df[column].isin(valid_groups)]

    return tts(filtered_df, stratify=filtered_df[column], train_size=train_size)


def main():
    '''takes raw datasets of reviews and metadata from amazon'''

    if len(sys.argv) != 2:
        print('usage: python process_data.py <domain>')
        sys.exit(1)

    domain = sys.argv[1]

    reviews_path = f'{domain}/{domain}.json'
    meta_path = f'{domain}/meta_{domain}.json'
    reviews_df = process_reviews(reviews_path)
    meta_df = process_metadata(meta_path)
    meta_df, reviews_df = sync(meta_df, reviews_df, n=5)
    reviews_df.to_csv(f'{domain}/reviews_text.tsv', sep='\t', index=False)
    meta_df.to_csv(f'{domain}/meta_synced.tsv', sep='\t', index=False)
    print('users:', reviews_df.user_id.nunique())
    print('items:', reviews_df.asin.nunique())

    (
        pd.melt(meta_df, id_vars=['asin'])
        .rename(columns={'variable': 'relation', 'value': 'attribute'})
        .to_csv(
            f'{domain}/kg_readable.tsv',
            sep='\t',
            index=False,
            quoting=2,
            escapechar='"',
        )
    )

    train, test = train_test_split(reviews_df)
    train.to_csv(f'{domain}/train.tsv', sep='\t', index=False)
    test.to_csv(f'{domain}/test.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
