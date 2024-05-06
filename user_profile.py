import argparse
import json
import random

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from llama import call_pipe, clean_text_series, get_pipe, num_tokens, timeit

pd.options.display.width = 0

template = '<Item Title>: "{}"; <Item Description>: "{}"; <User Review> "{}"; <User Score>: {}'


def cut_to(sentence, tokenizer, max_length):
    return tokenizer.convert_tokens_to_string(tokenizer.tokenize(sentence)[: max_length])


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
        help='maximum number of reviews per user'
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--prompt_path', type=str, default='prompts/prompts_proper.json')
    args = parser.parse_args()
    if 'Llama-2' in args.model_name:
        args.max_length = 4096
    elif 'Llama-3' in args.model_name:
        args.max_length = 8192
    else:
        raise ValueError(f'model not llama: {args.model_name}')
    return args


@timeit
def load_data(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    reviews = pd.read_table(f'{args.data}/reviews_text.tsv')
    meta = pd.read_table(f'{args.data}/meta_synced.tsv')

    train = pd.read_table(f'{args.data}/reshuffle_{args.seed}/train.tsv')
    test = pd.read_table(f'{args.data}/reshuffle_{args.seed}/test.tsv')

    # users = set(random.sample(list(test.user_id.unique()), 5))
    users = set(train.user_id.unique()) | set(test.user_id.unique())

    reviews = reviews[reviews.user_id.isin(users)]
    reviews['review'] = clean_text_series(reviews.review).apply(cut_to, args=(tokenizer, args.max_review_length))
    meta['title'] = clean_text_series(meta.title)
    meta['description'] = clean_text_series(meta.description).apply(cut_to, args=(tokenizer, args.max_review_length))

    train = train[train.user_id.isin(users)].groupby('user_id').head(args.max_review_number)  # only take max_review_number reviews
    test = test[test.user_id.isin(users)]

    train = train.merge(meta, on='asin')
    test = test.merge(meta, on='asin')
    train = train.merge(reviews, on=['user_id', 'asin', 'rating'])
    test = test.merge(reviews, on=['user_id', 'asin', 'rating'])

    return train, test, meta.set_index('asin')


def downsample(reviews, meta, n=10):
    vc = reviews.user_id.value_counts()
    reviews = reviews[(reviews.user_id.isin(vc[vc > 4].index)) & (reviews.user_id.isin(vc[vc < 10].index))]
    users = random.sample(list(reviews.user_id.unique()), n)
    reviews = reviews[reviews.user_id.isin(users)]
    meta = meta[meta.asin.isin(set(reviews.asin.unique()))]
    return reviews, meta


@timeit
def generate_profile(pipe, data, args):
    with open(args.prompt_path, 'r') as file:
        prompt = pipe.tokenizer.apply_chat_template(json.load(file)['user_profile'], tokenize=False, add_generation_prompt=True)
        prompt_len = num_tokens(prompt, pipe.tokenizer)

    if 'Llama-3' in args.model_name:
        profile_path = f'profiles_{args.temp}_llama3.tsv'
    elif 'Llama-2' in args.model_name:
        profile_path = f'profiles_{args.temp}_llama2.tsv'
    else:
        raise ValueError(f'model not llama: {args.model_name}')

    # if os.path.exists(profile_path):
    if False:
        descriptions = pd.read_table(profile_path)
    else:
        descriptions = pd.DataFrame()
        for user, group in data.groupby('user_id'):
            filler = ''

            # control length of final prompt by breaking early
            for ind, (_, row) in enumerate(group.iterrows(), start=1):
                filler += f'{ind}. ' + template.format(row['title'], row['description'], row['review'], row['rating']) + '\n'
                if prompt_len + num_tokens(filler, pipe.tokenizer) > args.max_length - args.max_new_tokens:
                    break
            descriptions.loc[user, 'prompt_profile'] = prompt.format(filler)

        descriptions.to_csv(profile_path, sep='\t', index=False)

    descriptions['profile'] = call_pipe(
        pipe=pipe,
        queries=descriptions['prompt_profile'].tolist(),
        batch_size=args.batch_size,
    )

    descriptions = descriptions.drop('prompt_profile', axis=1)
    descriptions.to_csv(profile_path, sep='\t')
    return descriptions


@timeit
def rate_items(pipe, profiles, data, args):
    with open(args.prompt_path, 'r') as file:
        prompt = pipe.tokenizer.apply_chat_template(json.load(file)['rating'], tokenize=False, add_generation_prompt=True)

    if 'Llama-3' in args.model_name:
        rating_path = f'rating_responses_{args.temp}_llama3.tsv'
    elif 'Llama-2' in args.model_name:
        rating_path = f'rating_responses_{args.temp}_llama2.tsv'
    else:
        raise ValueError(f'model not llama: {args.model_name}')

    data['profile'] = data['user_id'].map(profiles['profile'])
    data['prompt_inference'] = prompt
    data['prompt_inference'] = data.apply(lambda x: x['prompt_inference'].format(x['profile'], x['title'], x['description']), axis=1)
    data.to_csv(rating_path, sep='\t', index=False)
    data['response'] = call_pipe(pipe=pipe, queries=data['prompt_inference'].tolist(), batch_size=args.batch_size)
    data['pred_score'] = data['response'].apply(lambda x: x.split("Final score:")[1] if 'Final score:' in x else np.nan)
    data.drop('prompt_inference', axis=1).to_csv(rating_path, sep='\t', index=False)


def rank_items(pipe, profiles, meta, args):
    with open(args.prompt_path, 'r') as file:
        prompt = pipe.tokenizer.apply_chat_template(json.load(file)['ranking'], tokenize=False, add_generation_prompt=True)

    ranking_path = 'reranking_responses.tsv'

    model = load_lightgcn(args)
    prediction = model.predict(users=profiles.index, top_n=50)
    profiles['y_pred'] = [[model.item_mapping_dict[i] for i in row] for row in prediction]

    def compose_rerank_prompt(row):
        items = []
        for ind, item in enumerate(row['y_pred'], start=1):
            items.append(f'{ind}. Title: \"{meta.loc[item, "title"]}\". Description: \"{meta.loc[item, "description"]}\"')
        return prompt.format(profiles.loc[row.name, 'profile'], '\n'.join(items))

    profiles['rerank_prompt'] = profiles.apply(compose_rerank_prompt, axis=1)
    profiles.to_csv(ranking_path, sep='\t', index=False)
    print('rerank prompts lens:', profiles['rerank_prompt'].apply(num_tokens, args=(pipe.tokenizer,)).tolist(), flush=True)

    profiles['rerank_response'] = call_pipe(pipe=pipe, queries=profiles['rerank_prompt'].tolist(), batch_size=args.batch_size)
    profiles = profiles.drop('rerank_prompt', axis=1)
    profiles.to_csv(ranking_path, sep='\t', index=False)


def main():
    args = parse_args()
    random.seed(args.seed)
    pipe = get_pipe(args, quantization='bfloat16')
    train, test, meta = load_data(args)

    profiles = generate_profile(pipe, train, args)

    # rate_items(pipe, profiles, test, args)
    # rank_items(pipe, profiles, meta, args)
    # infer_scores(pipe, test, profiles, args)


if __name__ == '__main__':
    main()
