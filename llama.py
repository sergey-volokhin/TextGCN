import argparse
import html
import json
import re
import string
import time
from functools import wraps
from unicodedata import normalize

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig, pipeline
from unidecode import unidecode


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
    parser.add_argument('--prompt_path', type=str, default='prompts/prompts_proper.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument(
        '--max_review_length',
        type=int,
        default=512,
        help='max length of a single review',
    )
    parser.add_argument(
        '--max_desc_len',
        type=int,
        default=1024,
        help='max length of a single description',
    )
    parser.add_argument('--no_reviews', action='store_true')
    parser.add_argument('--reviews', action='store_true')
    args = parser.parse_args()
    args.max_length = {
        'llama_2': 4096,
        'llama_3': 8192,
    }[args.model_name.lower()]

    assert args.no_reviews or args.reviews, 'either --no_reviews or --reviews must be set'
    return args


''' string stuff '''
unprintable_pattern = re.compile(f'[^{re.escape(string.printable)}]')

# default NA values from pd.read_csv
na_values = [
    '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',  #
    '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',  #
    'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null', 'none', 'None',
]
na_values_dict = {value: np.nan for value in na_values}  # Use a dict for replace

emoji_pattern = re.compile(
    u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0"
    u"\U00002702-\U000027B0\U000024C2-\U0001F251\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+",
    flags=re.UNICODE,
)

default_hf_kwargs = {
    'max_new_tokens': 512,
    'min_new_tokens': 10,
    'temperature': 0.4,
    'do_sample': True,
    'return_full_text': False,
    'repetition_penalty': 1.2,
}

longer_names = {'llama_2': 'meta-llama/Llama-2-7b-chat-hf', 'llama_3': 'meta-llama/Meta-Llama-3-8B-Instruct'}


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f'{func.__name__:<30} {time.perf_counter() - start:>6.2f} sec', flush=True)
        return res
    return wrapper


def sort_process_unsort(func):
    '''Decorator to sort the input by char length, process it, and then unsort the output'''
    @wraps(func)
    def wrapper(queries, **kwargs):
        sorted_strings_with_indices = sorted(enumerate(queries), key=lambda x: -len(x[1]))
        sorted_strings = [string for _, string in sorted_strings_with_indices]
        processed_values = func(queries=sorted_strings, **kwargs)
        unsorted_values = [None] * len(queries)
        for (original_index, _), value in zip(sorted_strings_with_indices, processed_values):
            unsorted_values[original_index] = value
        return unsorted_values
    return wrapper


def clean_text(text):
    # Combine operations to reduce the number of apply calls
    text = unidecode(html.unescape(normalize('NFKD', text)))
    text = BeautifulSoup(text, "html.parser").get_text()  # Removes all HTML tags
    text = emoji_pattern.sub('', text)
    text = unprintable_pattern.sub('', text)
    text = re.sub('[\s_]+', ' ', text)  # Remove extra whitespaces
    return text


def clean_text_series(series):
    cleaned_series = series.apply(clean_text).str.lstrip(string.punctuation + string.whitespace)
    cleaned_series.replace(na_values_dict, inplace=True)  # Efficient NA replacement
    return cleaned_series[cleaned_series.str.len() > 1]


def cut_to(sentence, tokenizer, max_length):
    return tokenizer.convert_tokens_to_string(tokenizer.tokenize(sentence)[:max_length])


@timeit
def get_data(args, tokenizer):
    '''
    kg_readable_w_gen_desc_v1       - generated with only torso items (deprecated)
    kg_readable_v2                  - added generated for tail items
    kg_readable_v3_w_reviews        - append reviews to prompts to generate texts
    kg_readable_v4_proper           - using proper prompts
    kg_readable_v4_proper_w_reviews - using proper prompts and reviews
    kg_readable_v5_temp_llama_x     - prompts are conversations (list of dict) instead of string

    if v1 exists, read it, drop items that are already generated
    concatenate it at the end

    meta: pd.DataFrame(index: asin, columns: ['title', 'description'])
    '''

    reviews = pd.read_table(f'{args.data}/reviews_text.tsv').sort_values('time')
    meta = pd.read_table(f'{args.data}/meta_synced.tsv').set_index('asin')[['description', 'title']]
    meta = meta[meta.index.isin(set(reviews.asin.unique()))]
    meta['title'] = clean_text_series(meta['title'])
    meta['description'] = clean_text_series(meta['description']).apply(cut_to, args=(tokenizer, args.max_desc_len))
    return meta


@timeit
def get_pipe(args, quantization='bfloat16'):

    if torch.cuda.is_available():
        model_kwargs = {'attn_implementation': "flash_attention_2"}
    else:
        print('ERROR: NO CUDA')
        model_kwargs = {}

    if quantization == '4bit':
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs['torch_dtype'] = torch.bfloat16
    elif quantization == 'bfloat16':
        model_kwargs['torch_dtype'] = torch.bfloat16
    elif quantization == 'float16':
        model_kwargs['torch_dtype'] = torch.float16
    else:
        raise ValueError(f'Unknown quantization: {quantization}')

    pipe = pipeline(
        task='text-generation',
        model=longer_names[args.model_name],
        model_kwargs=model_kwargs,
        max_length=args.max_length,
        min_new_tokens=args.min_new_tokens,
        device_map='auto',
        temperature=args.temp,
        do_sample=True,
        return_full_text=False,
        repetition_penalty=1.2,
        truncation=True,
    )
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id  # so we can do batching
    # pipe.tokenizer.pad_token = "[PAD]"
    pipe.tokenizer.padding_side = "left"
    return pipe


def get_prompt_v1(row, prompt, tokenizer):
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    ).format(row['title'], row['description'])


def get_prompt_v2(row, prompt):
    ''' not applying chat template directly '''
    return [{k: v.format(row['title'], row['description']) for k, v in utterance.items()} for utterance in prompt]


# def get_prompt_w_reviews(row, prompt_type):
#     prompt = f'''Title: "{row['title']}"\n'''
#     if row['description'] is not np.nan:
#         prompt += f'''Seller Description: "{row['description']}"\n'''
#     prompt += f'''Customer reviews:\n"{row['reviews']}"\n\n'''
#     return new_prompts[prompt_type] + prompt


@timeit
@sort_process_unsort
def call_pipe(pipe, queries, batch_size):
    terminators = [pipe.tokenizer.eos_token_id]
    if 'llama-3' in pipe.model.config._name_or_path.lower():
        terminators.append(pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    return [
        i[0]['generated_text'].lstrip('.!?, \n\r\t-').strip(' ,\n\t-')
        for i in pipe(queries, batch_size=batch_size, eos_token_id=terminators)
    ]


def call_pipe_dummy(pipe, queries, batch_size):
    return [f'dummy response to {query}' for query in queries]


def save(df, path):
    df.reset_index().to_csv(path, sep='\t', index=False)


def num_tokens(text, tokenizer):
    return len(tokenizer(text)['input_ids']) if isinstance(text, str) else np.nan


def num_tokens_conversation(conversation, tokenizer):
    return num_tokens('\n'.join([turn['content'] for turn in conversation]), tokenizer)


def process_type(df, p_type, prompt, pipe, args):
    '''
    process a single prompt type:
    - generate prompts
    - generate text
    - try to fill missing values by regenerating
    '''

    # generate prompts
    df[f'prompt_{p_type}'] = df.apply(get_prompt_v2, args=(prompt,), axis=1)

    # drop too long prompts
    df['tokens'] = df[f'prompt_{p_type}'].apply(num_tokens_conversation, args=(pipe.tokenizer,))
    df = df[df['tokens'] < args.max_length - args.min_new_tokens]
    print('after removing too long', len(df), flush=True)

    # generate text
    df[f'gen_{p_type}'] = call_pipe(pipe=pipe, queries=df[f'prompt_{p_type}'].tolist(), batch_size=args.batch_size)

    # redo missing or too short
    df['tokens'] = df[f'gen_{p_type}'].apply(num_tokens, args=(pipe.tokenizer,))
    missing = df[(df[f'gen_{p_type}'].isna()) | (df['tokens'] < args.min_new_tokens)][f'prompt_{p_type}']
    if not missing.empty:
        print(f'filling {len(missing)} missing values for {p_type}', flush=True)
        generated = [pipe(i)[0]['generated_text'] for i in tqdm(missing.values, desc='fill missing')]
        df.loc[missing.index, f'gen_{p_type}'] = generated

    print(f'done generating {p_type} prompts for {len(df)} items', flush=True)
    return df


def add_reviews(df, p_type, pipe, args):
    # generate prompts

    reviews = pd.read_table(f'{args.data}/reviews_text.tsv')
    reviews['tokens'] = reviews.review.apply(num_tokens, args=(pipe.tokenizer,))
    reviews = reviews[reviews.tokens < args.max_review_length]
    reviews = reviews.sort_values('time').groupby('asin').head(5).drop('tokens', axis=1)

    num_items = len(df)
    df['reviews'] = (
        reviews.groupby('asin')
        .review.apply(
            lambda x: ('Review {}: ' + '\nReview {}: '.join(x.str.replace('{', '{{').str.replace('}', '}}'))).format(
                *range(1, len(x) + 1)
            )
        )
        .reindex(df.index)
    )

    df[f'prompt_{p_type}_w_reviews'] = df.apply(get_prompt_w_reviews, args=(p_type,), axis=1)

    # drop too long prompts
    df['tokens'] = df[f'prompt_{p_type}_w_reviews'].apply(num_tokens, args=(pipe.tokenizer,))
    df = df[df['tokens'] < args.max_length - args.min_new_tokens]
    print('after removing too long', len(df), flush=True)

    # generate text
    df[f'gen_{p_type}_w_reviews'] = call_pipe(
        pipe=pipe,
        queries=df[f'prompt_{p_type}_w_reviews'].tolist(),
        batch_size=args.batch_size,
    )

    # redo missing or too short
    df['tokens'] = df[f'gen_{p_type}_w_reviews'].apply(num_tokens, args=(pipe.tokenizer,))
    missing = df[(df[f'gen_{p_type}_w_reviews'].isna()) | (df['tokens'] < args.min_new_tokens)][
        f'prompt_{p_type}_w_reviews'
    ]
    print(f'filling {len(missing)} missing values for {p_type}', flush=True)
    generated = [pipe(i)[0]['generated_text'] for i in tqdm(missing.values)]
    df.loc[missing.index, f'gen_{p_type}_w_reviews'] = generated

    df['tokens'] = df[f'prompt_{p_type}_w_reviews'].apply(num_tokens, args=(pipe.tokenizer,))
    df = df[df['tokens'] > args.min_new_tokens].drop(columns='tokens')
    print(f'done generating {p_type} prompts, {len(df)} items left out of initial {num_items}', flush=True)
    return df


def main():

    args = parse_args()
    with open(args.prompt_path, 'r') as file:
        prompts = json.load(file)['item_prompts']

    pipe = get_pipe(args, quantization='bfloat16')
    meta = get_data(args, pipe.tokenizer)
    print(f'generate text for {len(meta)} items', flush=True)

    if args.no_reviews:
        for prompt_type, prompt in prompts.items():
            meta = process_type(meta, prompt_type, prompt, pipe, args)
            save(meta, f'{args.data}/kg_readable_v5_{args.temp}_{args.model_name}.tsv')

    # if args.reviews:
    #     for prompt_type in prompts:
    #         meta = add_reviews(meta, prompt_type, pipe, args)
    #         save(meta, f'{args.data}/kg_readable_v4_proper_w_reviews_{args.model_name.split("/")[-1]}.tsv')


if __name__ == '__main__':
    main()
