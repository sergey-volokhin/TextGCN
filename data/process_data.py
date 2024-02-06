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
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split as tts
from tqdm.auto import tqdm
from unidecode import unidecode

unprintable_pattern = re.compile(f'[^{re.escape(string.printable)}]')

# default NA values from pd.read_csv
na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
             '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
             'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'none', 'None']
na_values_dict = {value: np.nan for value in na_values}  # Use a dict for replace

meta_required_fields = ['asin', 'title']
meta_acceptable_fields = ['title', 'description', 'asin', 'category', 'main_cat']

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


def lines_in_file(path):
    return int(subprocess.run(['wc', '-l', path], stdout=subprocess.PIPE).stdout.split()[0])


def clean_text(text):
    # Combine operations to reduce the number of apply calls
    text = unidecode(html.unescape(normalize('NFKD', text)))
    text = BeautifulSoup(text, "html.parser").get_text()  # Removes all HTML tags
    text = emoji_pattern.sub('', text)
    text = unprintable_pattern.sub('', text)
    text = re.sub('[\s_]+', ' ', text)
    return text


@timeit
def clean_text_series(series):
    cleaned_series = series.apply(clean_text).str.lstrip(string.punctuation + string.whitespace)
    cleaned_series.replace(na_values_dict, inplace=True)  # Efficient NA replacement
    return cleaned_series[cleaned_series.str.len() > 1]


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
    with open(path, 'r') as file:
        for row in tqdm(file, desc='processing metadata', dynamic_ncols=True, total=lines_in_file(path)):
            if not row:
                continue
            row = json.loads(row)
            if all(i in row for i in meta_required_fields):
                cleaned.append({k: row[k] for k in meta_acceptable_fields})
    df = pd.DataFrame(cleaned).drop_duplicates('asin')
    df['description'] = clean_text_series(df['description'].apply(' '.join))
    df['title'] = clean_text_series(df['title'])
    return df.replace(na_values, np.nan).dropna().reset_index(drop=True)


@timeit
def process_reviews(path):
    '''
    remove all unused fields from reviews,
    normalize textual fields
    '''
    fields = ['reviewText', 'reviewerID', 'asin', 'unixReviewTime', 'overall']
    cleaned = []
    with open(path, 'r') as file:
        for row in tqdm(file, desc='proc reviews', dynamic_ncols=True, total=lines_in_file(path)):
            if not row:
                continue
            row = json.loads(row)
            if all(i in row for i in fields):
                cleaned.append({k: row[k] for k in fields})
    df = core_n(
        pd.DataFrame(cleaned)
        .rename(columns={
            'reviewerID': 'user_id',
            'reviewText': 'review',
            'unixReviewTime': 'time',
            'overall': 'rating'})
        .drop_duplicates(subset=['user_id', 'asin'])
        .replace(na_values, np.nan)
        .dropna()
        .astype({'rating': int}),
        columns=('user_id',),
        n=2,
    )
    df = df[df.rating.isin(range(1, 6))]
    df['review'] = clean_text_series(df.review)
    return df.replace(na_values, np.nan).dropna().reset_index(drop=True)


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
def not_boring(df, columns=('user_id', 'asin')):
    ''' remove all user and items whose reviews have the same rating (e.g. all 5s) '''
    if len(columns) == 1:
        ratings = df.groupby(columns[0])['rating'].nunique()
        return df[df[columns[0]].isin(ratings[ratings > 1].index)]
    while True:
        shape = df.shape
        for column in columns:
            ratings = df.groupby(column)['rating'].nunique()
            df = df[df[column].isin(ratings[ratings > 1].index)]
        if df.shape == shape:
            return df


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


def main(seed=42):
    '''takes raw datasets of reviews and metadata from amazon'''

    if len(sys.argv) == 1:
        print('usage: python process_data.py <domain>')
        sys.exit(1)

    domain = sys.argv[1]
    seed = sys.argv[2] if len(sys.argv) > 2 else 42

    read_folder = '../../amazon_data'
    write_folder = f'rgcl_sampled/many_{domain}_new/textgcn'
    os.makedirs(write_folder, exist_ok=True)

    reviews_path = f'{read_folder}/{domain}/{domain}.json'
    meta_path = f'{read_folder}/{domain}/meta_{domain}.json'
    reviews_df = process_reviews(reviews_path)
    meta_df = process_metadata(meta_path)
    meta_df, reviews_df = sync(meta_df, reviews_df, n=2)
    reviews_df.to_csv(f'{write_folder}/reviews_text.tsv', sep='\t', index=False)
    print('users:', reviews_df.user_id.nunique())
    print('items:', reviews_df.asin.nunique())

    (
        pd.melt(meta_df, id_vars=['asin'])
        .rename(columns={'variable': 'relation', 'value': 'attribute'})
        .to_csv(
            f'{write_folder}/kg_readable.tsv',
            sep='\t',
            index=False,
            quoting=2,
            escapechar='"',
        )
    )
    # reviews_df = pd.read_table(f'{domain}/reviews_text.tsv', sep='\t')
    # train, test = train_test_split(df=reviews_df.drop(columns=['review']), seed=seed)
    # train.to_csv(f'{domain}/train.tsv', sep='\t', index=False)
    # test.to_csv(f'{domain}/test.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()

    # for domain in ['Digital_Music', 'Movies_and_TV', 'Toys_and_Games', 'Electronics', 'Books']:
    # domain = 'Movies_and_TV'
    # main(domain)
