import numpy as np
import argparse
import json
import os
import random

import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm


from llama import call_pipe, clean_text_series, get_pipe, num_tokens, timeit

pd.options.display.width = 0

template = '<Item Title>: "{}"; <Item Description>: "{}"; <User Review> "{}"; <User Score>: {}'


def cut_to(sentence, tokenizer, max_length):
    return tokenizer.convert_tokens_to_string(tokenizer.tokenize(sentence)[:max_length])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--data', '-d', type=str, default='data/sampled_Toys_and_Games')
    parser.add_argument('--temp', type=float, default=0.01)
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
    parser.add_argument('--prompt_path', type=str, default='prompts/prompts_proper.json')
    parser.add_argument('--regenerate', action='store_true', help='regenerate prompts')
    args = parser.parse_args()
    if 'llama-2' in args.model_name.lower():
        args.max_length = 4096
    elif 'llama-3' in args.model_name.lower():
        # args.max_length = 4096
        args.max_length = 8192
    else:
        raise ValueError(f'model not llama: {args.model_name}')
    return args


@timeit
def clean_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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

    train = pd.read_table(f'{args.data}/reshuffle_{args.seed}/train.tsv')
    test = pd.read_table(f'{args.data}/reshuffle_{args.seed}/test.tsv')

    users = set(train.user_id.unique()) | set(test.user_id.unique())
    reviews = reviews[reviews.user_id.isin(users)]

    train = train[train.user_id.isin(users)].groupby('user_id').head(args.max_review_number)

    return train.merge(meta, on='asin').merge(reviews, on=['user_id', 'asin', 'rating'])


def process_group(group, prompt_len, args, prompt, tokenizer):
    filler = ''
    current_len = prompt_len
    for ind, row in enumerate(group.itertuples(), start=1):
        # control length of the final prompt by breaking early
        to_add = f'{ind}. ' + template.format(row.title, row.description, row.review, row.rating) + '\n'
        filler += to_add
        current_len += num_tokens(to_add, tokenizer)
        if current_len > args.max_length - args.max_new_tokens:
            break
    return [{k: v.format(filler) if k == 'content' else v for k, v in p.items()} for p in prompt]


def load_cache(path):
    if os.path.exists(os.path.join(path, 'embeddings/profiles_cache.tsv')):
        return (
            pd.read_table(os.path.join(path, 'embeddings/profiles_cache.tsv'))
            .set_index('prompt_profile')['profile']
            .to_dict()
        )
    return {}


def load_prompt(path, tokenizer):
    with open(path, 'r') as file:
        prompt = json.load(file)['user_profile']
    prompt_length = sum(
        num_tokens(value, tokenizer)
        for turn in prompt
        for key, value in turn.items()
        if key == 'content'
    )
    return prompt, prompt_length


@timeit
def generate_profile(pipe, tokenizer, data, args):

    prompt, prompt_length = load_prompt(args.prompt_path, tokenizer)
    cache = load_cache(args.data)

    if 'llama-3' in args.model_name.lower():
        profile_path = f'{args.data}/reshuffle_{args.seed}/profiles_{args.temp}_llama3.tsv'
    elif 'llama-2' in args.model_name.lower():
        profile_path = f'{args.data}/reshuffle_{args.seed}/profiles_{args.temp}_llama2.tsv'
    else:
        raise ValueError(f'model not llama: {args.model_name}')

    # construct and fill prompts
    if os.path.exists(profile_path) and not args.regenerate:
        descriptions = pd.read_table(profile_path)
    else:
        descriptions = []
        for user, group in tqdm(data.groupby('user_id'), dynamic_ncols=True):
            descriptions.append((user, process_group(group, prompt_length, args, prompt, tokenizer)))
        descriptions = pd.DataFrame(descriptions, columns=['user_id', 'prompt_profile']).set_index('user_id')
        descriptions.to_csv(profile_path, sep='\t')

    # select only the prompts that need to be generated
    descriptions['profile'] = descriptions['prompt_profile'].map(lambda x: cache.get(x, np.nan))
    to_generate = descriptions[descriptions['profile'].isna()].prompt_profile.tolist()
    print('len(to_generate)', len(to_generate), 'instead of', len(descriptions))
    generated = call_pipe(
        pipe=pipe,
        queries=to_generate,
        batch_size=args.batch_size,
    )
    descriptions['profile'] = descriptions['profile'].astype(object)  # dunno why this is needed
    descriptions.loc[descriptions['profile'].isna(), 'profile'] = generated
    descriptions.to_csv(profile_path, sep='\t', index=False)

    cache.update(descriptions.set_index('prompt_profile')['profile'].to_dict())
    pd.DataFrame(cache.items(), columns=['prompt_profile', 'profile']).to_csv(
        os.path.join(args.data, 'embeddings/profiles_cache.tsv'),
        sep='\t',
        index=False,
    )

    return descriptions


def main():
    args = parse_args()
    random.seed(args.seed)
    pipe = get_pipe(args, quantization='bfloat16')
    data = load_data(args)
    profiles = generate_profile(pipe, pipe.tokenizer, data, args)


if __name__ == '__main__':
    main()
