import html
import json
import re
import string
import numpy as np
from unicodedata import normalize
import os

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

printable = string.punctuation + string.ascii_letters + string.digits + ' '

# default NA values from pd.read_csv
na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
             '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
             'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']


def normalize_string(s):
    s = normalize("NFKD", deEmojify(re.sub('<[^<]+?>', '', html.unescape(unidecode(s)))))
    s = ''.join(filter(lambda x: x in printable, s))
    result = (
        s.replace('gl_', '')
        .replace('_display_on_website', '')
        .replace('_', ' ')
        .replace('-', ' ')
        .lower()
        .strip(' ~.,!?\n*-\'"`:@#$%^&()=+[]\\/')
    )
    return result if len(result) > 1 else ''


def deEmojify(text):
    '''shamelessly yoinked from stackoverflow'''
    if isinstance(text, float):
        return ''
    emoji_pattern = re.compile(
        u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0"
        u"\U00002702-\U000027B0\U000024C2-\U0001F251\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)


def process_metadata(metadata):
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
    fields = ['title', 'description', 'asin']
    cleaned = []
    for row in tqdm(metadata, desc='proc meta'):
        if not all(i in row for i in fields):
            continue
        row = json.loads(row)
        row['description'] = normalize_string(' '.join([i for i in row['description'] if i]))
        if row['description']:
            cleaned.append({k: row[k] for k in fields})
    return pd.DataFrame(cleaned).drop_duplicates()


def process_reviews(reviews):
    '''
    remove all unused fields from reviews,
    normalize textual fields

    returns: pd.DataFrame(dict(['user_id', 'review', 'asin', 'unixReviewTime'], Any))
    '''
    fields = ['reviewText', 'reviewerID', 'asin', 'unixReviewTime']
    cleaned = []
    for row in tqdm(reviews, desc='proc reviews'):
        if not row:
            continue
        row = json.loads(row)
        if not all(i in row for i in fields):
            continue
        row['reviewText'] = normalize_string(row['reviewText'])
        if row['reviewText']:
            cleaned.append({k: row[k] for k in fields})
    return (
        pd.DataFrame(cleaned)
        .rename(columns={'reviewerID': 'user_id', 'reviewText': 'review', 'unixReviewTime': 'time'})
        .drop_duplicates(subset=['user_id', 'asin'])
        .replace(na_values, np.nan)
        .dropna()
    )


def intersect(df1, df2, column='asin'):
    ''' remove all items that don't appear in both dataframes '''
    intersection = set(df1[column].unique()).intersection(set(df2[column].unique()))
    df1 = df1[df1[column].isin(intersection)]
    df2 = df2[df2[column].isin(intersection)]
    return df1, df2


def core_n(reviews, n=5, columns=('asin', 'user_id')):
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


def sync(meta, reviews, n=1):
    if n == 1:
        return intersect(meta, reviews)
    while True:
        r_shape = reviews.shape
        m_shape = meta.shape
        reviews = core_n(reviews, n)
        meta, reviews = intersect(meta, reviews)
        if reviews.shape == r_shape and meta.shape == m_shape:
            return meta, reviews


def train_test_split(df: pd.DataFrame, column='user_id', train_size=0.8):
    ''' split df into train and test, ensuring that all users are in both sets '''
    train, test = [], []
    for _, group in tqdm(df.groupby(column)):
        if len(group) < 3:
            continue
        group = group.sample(frac=1, random_state=0)
        group_train_size = round(group.shape[0] * train_size)
        train.append(group.iloc[:group_train_size])
        test.append(group.iloc[group_train_size:])
    train = pd.concat(train)
    test = pd.concat(test)
    test = test[test['asin'].isin(train['asin'].unique())]
    return train, test


def main():
    ''' takes raw datasets of reviews and metadata from amazon '''

    for domain in ['Electronics', 'Toys_and_Games', 'Movies_and_TV']:

        reviews_df = process_reviews(open(f'{domain}/{domain}.json', 'r').read().split('\n'))
        meta_df = process_metadata(open(f'{domain}/meta_{domain}.json', 'r').read().split('\n'))

        meta_df, reviews_df = sync(meta_df, reviews_df, n=7)
        print(domain, reviews_df.user_id.nunique(), reviews_df.asin.nunique())

        reviews_df.to_csv(f'{domain}/reviews_synced.tsv', sep='\t', index=False)
        meta_df.to_csv(f'{domain}/meta_synced.tsv', sep='\t', index=False)

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
