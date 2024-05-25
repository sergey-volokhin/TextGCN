import argparse
import json
import os
import random

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from llama import (
    call_pipe,
    call_pipe_dummy,
    clean_text_series,
    cut_to,
    get_pipe,
    longer_names,
    num_tokens,
    num_tokens_conversation,
    timeit,
)

pd.options.display.width = 0

template = '<Item Title>: "{}"; <Item Description>: "{}"; <User Review> "{}"; <User Score>: {}'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_3', choices=['llama_2', 'llama_3'])
    parser.add_argument('--data', '-d', type=str, default='data/sampled_Toys_and_Games')
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument(
        '--min_new_tokens',
        type=int,
        default=10,
        help='min number of new tokens generated by the model',
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='max number of new tokens generated by the model',
    )
    parser.add_argument(
        '--max_review_length',
        type=int,
        default=200,
        help='max length of a single review',
    )
    parser.add_argument(
        '--max_review_number',
        type=int,
        default=30,
        help='maximum number of reviews per user',
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--prompts_template_path', type=str, default='prompts/prompts_proper.json')
    parser.add_argument('--regenerate', action='store_true', help='regenerate prompts')
    args = parser.parse_args()
    args.max_length = {
        'llama_2': 4096,
        'llama_3': 8192,
    }[args.model_name.lower()]
    return args


@timeit
def clean_data(args):
    tokenizer = AutoTokenizer.from_pretrained(longer_names[args.model_name])

    reviews = pd.read_table(f'{args.data}/reviews_text.tsv').dropna()
    reviews['review'] = clean_text_series(reviews.review).apply(cut_to, args=(tokenizer, args.max_review_length))
    reviews.to_csv(f'{args.data}/reviews_text_clean.tsv', sep='\t', index=False)

    meta = pd.read_table(f'{args.data}/meta_synced.tsv').dropna()
    meta['title'] = clean_text_series(meta.title)
    meta['description'] = clean_text_series(meta.description).apply(cut_to, args=(tokenizer, args.max_review_length))
    meta.to_csv(f'{args.data}/meta_synced_clean.tsv', sep='\t', index=False)

    return reviews, meta


def load_data(args):
    if os.path.exists(f'{args.data}/reviews_text_clean.tsv') and os.path.exists(f'{args.data}/meta_synced_clean.tsv'):
        reviews = pd.read_table(f'{args.data}/reviews_text_clean.tsv').dropna()
        meta = pd.read_table(f'{args.data}/meta_synced_clean.tsv').dropna()
    else:
        reviews, meta = clean_data(args)

    train = pd.read_table(f'{args.data}/reshuffle_{args.seed}/train_ranking.tsv')
    test = pd.read_table(f'{args.data}/reshuffle_{args.seed}/test_ranking.tsv')

    users = set(train.user_id.unique()) | set(test.user_id.unique())
    reviews = reviews[reviews.user_id.isin(users)]

    train = train[train.user_id.isin(users)].groupby('user_id').head(args.max_review_number)

    return train.merge(meta, on='asin').merge(reviews, on=['user_id', 'asin', 'rating'])


def process_group(group, prompt_length, max_prompt_length, prompt_template, tokenizer):
    '''append reviews one at a time until the prompt is too long'''
    filler = ''
    current_len = prompt_length
    for ind, row in enumerate(group.itertuples(), start=1):
        # control length of the final prompt by breaking early
        to_add = f'{ind}. ' + template.format(row.title, row.description, row.review, row.rating) + '\n'
        filler += to_add
        current_len += num_tokens(to_add, tokenizer)
        if current_len > max_prompt_length:
            break
    return str([{k: v.format(filler) if k == 'content' else v for k, v in p.items()} for p in prompt_template])


def load_cache(path):
    '''
    df with [prompt, profile]
    profile doesn't depend on model b.c. prompts are different
    '''
    if os.path.exists(path):
        return pd.read_table(path).set_index('prompt_profile')['profile'].to_dict()
    print('profile cache not found, creating')
    return {}


def load_prompt_template(path):
    with open(path, 'r') as file:
        return json.load(file)['user_prompts']['user_profile']


def construct_prompts(df, args, prompt_template, tokenizer):
    path = f'{args.data}/reshuffle_{args.seed}/prompts_{args.model_name}.tsv'  # prompts don't depend on temp

    if os.path.exists(path) and not args.regenerate:
        return pd.read_table(path, usecols=['user_id', 'prompt_profile'])

    prompts_df = []
    prompt_length = num_tokens_conversation(prompt_template, tokenizer)
    for user, group in tqdm(df.groupby('user_id'), dynamic_ncols=True):
        prompts_df.append(
            (
                user,
                process_group(
                    group,
                    prompt_length=prompt_length,
                    max_prompt_length=args.max_length - args.max_new_tokens,
                    prompt_template=prompt_template,
                    tokenizer=tokenizer,
                ),
            )
        )
    prompts_df = pd.DataFrame(prompts_df, columns=['user_id', 'prompt_profile']).set_index('user_id')
    print(f'saving filled prompts into {path}')
    # prompts_df.to_csv(path, sep='\t')
    return prompts_df


@timeit
def generate_profile(data, args, pipe):
    '''
    profiles file format:            v each model has own column
        user_id \t prompt_profile \t profile_{model_name}
                   ^ list of dicts so the same for all models
    '''

    profile_path = f'{args.data}/reshuffle_{args.seed}/profiles_{args.temp}_{args.model_name}.tsv'

    cache_path = f'{args.data}/profiles_cache_{args.temp}_{args.model_name}.tsv'
    cache = load_cache(cache_path)

    prompt_template = load_prompt_template(args.prompts_template_path)
    prompts_df = construct_prompts(data, args, prompt_template, pipe.tokenizer)

    # select only the prompts that need to be generated
    prompts_df['profile'] = prompts_df['prompt_profile'].map(cache)
    to_generate = prompts_df[prompts_df['profile'].isna()].prompt_profile.apply(eval).tolist()
    print(f'caching saved us {100 * (1 - len(to_generate) / len(prompts_df)):.1f}%, generating for {len(to_generate)}')
    # generate the prompts and put back into dataframe
    generated = call_pipe(
        pipe=pipe,
        queries=to_generate,
        batch_size=args.batch_size,
    )
    prompts_df['profile'] = prompts_df['profile'].astype(object)  # dunno why this is needed
    prompts_df.loc[prompts_df['profile'].isna(), 'profile'] = generated
    prompts_df.to_csv(profile_path, sep='\t', index=False)
    print(f'save profiles into {profile_path}')

    # update cache
    cache.update(prompts_df.set_index('prompt_profile')['profile'].to_dict())
    pd.DataFrame(cache.items(), columns=['prompt_profile', 'profile']).to_csv(cache_path, sep='\t', index=False)
    print(f'saving cached profiles into {cache_path}')

    return prompts_df


def main():
    args = parse_args()
    random.seed(args.seed)
    pipe = get_pipe(args, quantization='bfloat16')
    data = load_data(args)
    profiles = generate_profile(data, args, pipe)


if __name__ == '__main__':
    main()
