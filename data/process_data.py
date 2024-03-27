import gzip
import html
import os
import re
import string
import subprocess
import sys
import time
from functools import wraps
from unicodedata import normalize

import numpy as np
import orjson as json
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from tqdm.auto import tqdm
from unidecode import unidecode

unprintable_pattern = re.compile(f'[^{re.escape(string.printable)}]')

# default NA values from pd.read_csv, they are detected when dataframe is read, but not when it is created
na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',  #
             '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',  #
             'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'none', 'None']
na_values_dict = {value: np.nan for value in na_values}  # Use a dict for replace
meta_required_fields = ['asin', 'title']

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
        print(f'{func.__name__:<30} {time.perf_counter() - start:>6.2f} sec')
        return res
    return wrapper


def clean_text_string(s):
    if not isinstance(s, str):
        return ''

    def remove_html_tags(text):
        return re.sub(r'<[^<]+?>', '', text)

    s = unidecode(s)
    s = html.unescape(s)
    s = normalize('NFKD', s)
    s = remove_html_tags(s)
    s = emoji_pattern.sub(r'', s)
    s = re.sub(unprintable_pattern, '', s)
    s = re.sub('[\s_]+', ' ', s)  # multiple whitespaces and underscores to single space
    s = s.lstrip(string.punctuation + string.whitespace)
    return s if len(s) > 5 else ''


def lines_in_file(path):
    if path.endswith('.gz') or not os.path.exists(path):
        return 0
    return int(subprocess.run(['wc', '-l', path], stdout=subprocess.PIPE).stdout.split()[0])


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
    cleaned = []
    with open(path, 'r') if path.endswith('.json') else gzip.open(path, 'rb') as file:
        for row in tqdm(file, desc='proc metadata', dynamic_ncols=True, total=lines_in_file(path)):
            if not row:
                continue
            row = json.loads(row)
            if all(i in row for i in meta_required_fields):
                cleaned.append({k: row[k] for k in meta_required_fields})
    df = pd.DataFrame(cleaned).drop_duplicates('asin')
    df['description'] = df['description'].apply(' '.join).apply(clean_text_string)
    df['title'] = df['title'].apply(clean_text_string)
    return df.replace(na_values, np.nan).dropna().reset_index(drop=True)


@timeit
def process_reviews(path, available_asins):
    '''
    remove all unused fields from reviews,
    normalize textual fields
    '''
    columns = {
        'asin': 'asin',
        'reviewerID': 'user_id',
        'reviewText': 'review',
        'unixReviewTime': 'time',
        'overall': 'rating',
    }
    cleaned = []
    with open(path, 'r') if path.endswith('.json') else gzip.open(path, 'rb') as file:
        for row in tqdm(file, desc='read reviews', dynamic_ncols=True, total=lines_in_file(path)):
            if not row:
                continue
            row = json.loads(row)
            if all(i in row for i in columns) and row['asin'] in available_asins:
                cleaned.append({k: row[k] for k in columns})
    df = core_n(
        pd.DataFrame(cleaned)
        .rename(columns=columns)
        .drop_duplicates(subset=['user_id', 'asin'])
        .astype({'rating': int})
        .replace(na_values, np.nan)
        .dropna(),
    )
    df.review = df.review.apply(clean_text_string)
    df = df[df.rating.isin(range(1, 6))]
    return df.replace(na_values_dict).dropna().reset_index(drop=True)


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
    n: int = 2,
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
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ''' split df into train and test, ensuring that all users are in both sets '''

    # Pre-filter groups with less than 3 elements
    group_sizes = df.groupby(column).size()
    valid_groups = group_sizes[group_sizes >= 3].index
    filtered_df = df[df[column].isin(valid_groups)]

    return tts(filtered_df, stratify=filtered_df[column], train_size=train_size, random_state=seed)


def main():
    '''takes raw datasets of reviews and metadata from amazon'''

    if len(sys.argv) < 2:
        print('usage: python process_data.py <domain> [seed]')
        sys.exit(1)

    domain = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    meta_path = f'{domain}/meta_{domain}.json'
    reviews_path = f'{domain}/{domain}.json'

    meta_df = process_metadata(meta_path)
    reviews_df = process_reviews(reviews_path, available_asins=set(meta_df.asin.unique()))

    meta_df, reviews_df = sync(meta_df, reviews_df, n=5)
    meta_df.to_csv(f'{domain}/meta_synced.tsv', sep='\t', index=False)
    reviews_df.to_csv(f'{domain}/reviews_text.tsv', sep='\t', index=False)

    train, test = train_test_split(reviews_df, seed=seed)
    train.to_csv(f'{domain}/train.tsv', sep='\t', index=False)
    test.to_csv(f'{domain}/test.tsv', sep='\t', index=False)

    print(f'reviews: {reviews_df.shape[0]:>7}')
    print(f'users:   {reviews_df.user_id.nunique():>7}')
    print(f'items:   {reviews_df.asin.nunique():>7}')
    print(f'train:   {train.shape[0]:>7}')
    print(f'test:    {test.shape[0]:>7}')


if __name__ == '__main__':
    main()
