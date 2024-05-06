import argparse
import html
import os
import random
import re
import string
import time
from functools import wraps
from unicodedata import normalize

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from unidecode import unidecode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--domain', '-d', type=str, default='enriched_Toys')
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument(
        '--min_new_tokens',
        type=int,
        default=10,
        help='min number of new tokens generated by the model',
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1024,
        help='max number of new tokens generated by the model',
    )
    parser.add_argument(
        '--max_review_length',
        type=int,
        default=512,
        help='max length of a single review',
    )
    parser.add_argument(
        '--max_description_length',
        type=int,
        default=1024,
        help='max length of a single description',
    )
    parser.add_argument('--no_reviews', action='store_true')
    parser.add_argument('--reviews', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if 'Llama-2' in args.model_name:
        args.max_length = 4096
    elif 'Llama-3' in args.model_name:
        args.max_length = 8192
    else:
        raise ValueError(f'model not llama: {args.model_name}')

    args.folder = f'data/{args.domain}/'
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

''' prompts '''
prompts = {
    'description': "Using provided information and any internal knowledge about this product, generate a concise product description that strictly focuses on the key features and benefits. Do not include the provided title in your output. Avoid directly addressing me in your response or acknowledging the task; just present the description in a standalone format.\n\n",
    'usecases': "Using provided information and any internal knowledge about this product, formulate a description explaining the practical applications and usability of this product in real-world scenarios, using the title and seller's description as a reference. Do not include the provided title in your output. Avoid directly addressing me in your response or acknowledging the task; just present the description in a standalone format.\n\n",
    'expert': "Based on the above details, generate an opinion as if you are an expert in the relevant field for this product. Consider the features, quality, market position, and any other pertinent aspects. Your response should reflect a professional assessment, highlighting potential strengths, weaknesses, and the overall value proposition of the product from an expert's perspective.\n\n",
}

new_prompts = {
    'description': "Generate a detailed product description that concisely highlights the key features and benefits, focusing on the product's specifications, functionality, and user experiences that are most relevant for internal recommendation algorithms, based on the provided item title, seller's description, and customer reviews. Aim to identify and emphasize unique selling points (USPs) and differentiators that can assist in accurately categorizing and matching the product within the recommender system. Ensure the description is clear, informative, and directly relevant to an internal audience, encapsulating the essence of the product without marketing fluff or SEO considerations. Limit the description to 200 words, presenting it in a standalone format without directly addressing the task or including the provided title in the output.\n\n",
    'usecases': "Given the title, seller's description, and user reviews of this product, craft a comprehensive overview that highlights its key features and practical applications in real-world scenarios. Illustrate its use with creative examples and hypothetical situations where the product can provide significant value or solve common problems. Emphasize any unique benefits and address potential user concerns identified from the reviews. Be concise, limit the description to 200 words, presenting it in a standalone format. Refrain from including the product title directly and avoid any direct communication or task acknowledgment in your presentation.\n\n",
    'expert': '''Based on the item title, seller's description, and customer reviews provided, succinctly generate an expert opinion on this product, focusing on the most critical aspects within a 500-token limit. Your analysis should seamlessly integrate the following aspects:\n\n- The product's comparative advantage over its competitors, highlighting distinctive features or areas where it may fall short.\n- The primary target audience for this product and reasons for its suitability.\n- How the product aligns with current industry trends or innovations.\n- Insights into usability and user experience based on feedback.\n- Environmental and ethical considerations related to the product, if relevant.\n- Suggestions for improvement or potential enhancements for future versions.\n- A brief evaluation of the product's price in relation to its overall value and market position.\n\nPlease provide a balanced, expert-level assessment that is informative yet concise, aiding in a nuanced product understanding for recommendation purposes.\n\n''',
}

proper_prompts_no_reviews = {
    "description": [
        {
            "role": "system",
            "content": "You are a product recommendation assistant. Your primary task is to generate product description based on the seller given description, item's title, and any internal knowledge you have about that item. It should be done in a way that would help the ranking system recommend more relevant items to users."
        },
        {
            "role": "user",
            "content": """You are asked to summarize provided item description and its title.\n# Item Information\n<Title>: "{}"; <Description>: {}\n\n# Task Requirement\nNow, please, provide a concise and accurate description of the item, highlighting the key features and benefits, focusing on the product's specifications, functionality, and user experiences. Aim to identify and emphasize unique selling points (USPs) and differentiators that can assist in accurately categorizing and matching the product within the recommender system. Ensure the description is clear, informative, and directly relevant to an internal audience, encapsulating the essence of the product. Limit the description to 200 words. Refrain from including the product title directly and avoid any direct communication or task acknowledgment in your presentation."""
        }
    ],
    "usecases": [
        {
            "role": "system",
            "content": "You are a product recommendation assistant. Your primary task is to generate potential usecases for the product based on its description, title, and any internal knowledge you have about that item. It should be done in a way that would help the ranking system recommend more relevant items to users."
        },
        {
            "role": "user",
            "content": """You are asked to come up with potential use cases for the provided item based on its title and description.\n# Item Information\n<Title>: "{}"; <Description>: {}\n\n# Task Requirement\nNow, please craft a comprehensive overview that highlights item's key features and practical applications in real-world scenarios. Illustrate its use with creative examples and hypothetical situations where the product can provide significant value or solve common problems. Emphasize any unique benefits and address potential user concerns identified from the reviews. Be concise, limit the description to 200 words, presenting it in a standalone format. Refrain from including the product title directly and avoid any direct communication or task acknowledgment in your presentation."""
        }
    ],
    "expert": [
        {
            "role": "system",
            "content": "You are a product recommendation assistant. Your primary task is to generate expert opinions for the product based on its description, title, and any internal knowledge you have about that item. It should be done in a way that would help the ranking system recommend more relevant items to users."
        },
        {
            "role": "user",
            "content": """You are asked to provide an expert opinion on the product based on the given description and title.\n# Item Information\n<Title>: "{}"; <Description>: {}\n\n# Task Requirement\nNow, please generate an expert opinion on this product. Consider the features, quality, market position, and any other pertinent aspects. Your response should reflect a professional assessment, highlighting potential strengths, weaknesses, and the overall value proposition of the product from an expert's perspectivem. Refrain from including the product title directly and avoid any direct communication or task acknowledgment in your presentation."""
        }
    ],
}


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f'{func.__name__:<30} {time.perf_counter() - start:>6.2f} sec')
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


@timeit
def get_data(args, tokenizer):
    '''
    kg_readable_w_gen_desc_v1       - generated with only torso items (deprecated)
    kg_readable_v2                  - added generated for tail items
    kg_readable_v3_w_reviews        - append reviews to prompts to generate texts
    kg_readable_v4_proper           - using proper prompts
    ##kg_readable_v4_proper_w_reviews - using proper prompts and reviews

    if v1 exists, read it, drop items that are already generated
    concatenate it at the end

    meta: pd.DataFrame(index: asin, columns: ['title', 'description'])
    '''

    reviews = pd.read_table(f'{args.folder}/reviews_text.tsv').sort_values('time')
    # meta = pd.read_table(f'{args.folder}/kg_readable.tsv').pivot(index='asin', columns='relation', values='attribute')[['description', 'title']]
    meta = pd.read_table(f'{args.folder}/meta_synced.tsv').set_index('asin')[['description', 'title']]
    meta = meta[meta.index.isin(set(reviews.asin.unique()))]
    meta['title'] = clean_text_series(meta.title)
    meta['description'] = clean_text_series(meta.description)
    meta.description = meta.description.apply(
        lambda x: tokenizer.convert_tokens_to_string(tokenizer.tokenize(x)[: args.max_description_length])
    )
    return meta.head(80)

@timeit
def get_pipe(args, quantization=None):

    model_kwargs = {'attn_implementation': "flash_attention_2"}

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
        model=args.model_name,
        model_kwargs=model_kwargs,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        device_map='auto',
        temperature=args.temp,
        do_sample=True,
        return_full_text=False,
        top_p=0.9,
        repetition_penalty=1.2,
        truncation=True,
        padding=True,
    )
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id  # so we can do batching
    # pipe.tokenizer.pad_token = "[PAD]"
    pipe.tokenizer.padding_side = "left"
    return pipe


def get_prompt_v1(row, prompt_type, tokenizer):
    return tokenizer.apply_chat_template(proper_prompts_no_reviews[prompt_type], tokenize=False, add_generation_prompt=True).format(row['title'], row['description'])


# def get_prompt_v1(row, prompt_type):
#     prompt = f'''Title: "{row['title']}"\n'''
#     if row['description'] is not np.nan:
#         prompt += f'''Seller Description: "{row['description']}"\n'''
#     return prompt + prompts[prompt_type]


# def get_prompt_w_reviews(row, prompt_type):
#     prompt = f'''Title: "{row['title']}"\n'''
#     if row['description'] is not np.nan:
#         prompt += f'''Seller Description: "{row['description']}"\n'''
#     prompt += f'''Customer reviews:\n"{row['reviews']}"\n\n'''
#     return new_prompts[prompt_type] + prompt


@timeit
@sort_process_unsort
def call_pipe(pipe, queries, batch_size):

    if 'Llama-3' in pipe.model.config._name_or_path:
        terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    elif 'Llama-2' in pipe.model.config._name_or_path:
        terminators = [pipe.tokenizer.eos_token_id]
    else:
        raise ValueError(f'Unknown model: {pipe.model.config._name_or_path}')

    return [i[0]['generated_text'].lstrip('.!?, \n\r\t-').strip(' ,\n\t-') for i in pipe(queries, batch_size=batch_size, eos_token_id=terminators)]


@timeit
@sort_process_unsort
def call_pipe_with_profiling(pipe, queries: list[str], batch_size: int):
    # Initialize an empty list to store the results
    results = []

    # Define the profiler context manager, specifying the activities to profile
    # and the directory where the trace will be saved.
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,  # This flags enables memory profiling
        with_stack=True,
    ) as prof:

        # Record the function of interest. Here 'inference' is a custom label for clarity.
        with record_function("inference"):
            results = [
                i[0]['generated_text'].lstrip('.!?, \n\r\t-').strip(' ,\n\t-')
                for i in pipe(queries, batch_size=batch_size)
            ]

    # After exiting the context, the profiler has the performance data.
    # Here, we're printing out the results. You could also save them to a file.
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")  # Export the trace to a file for visualization

    return results


def save(meta, path):
    meta.reset_index().to_csv(path, sep='\t', index=False)


def num_tokens(text, tokenizer):
    return len(tokenizer(text)['input_ids']) if isinstance(text, str) else np.nan


def process_type(df, p_type, pipe, args):
    '''
    process a single prompt type:
    - generate prompts
    - generate text
    - try to fill missing values by regenerating
    '''

    # generate prompts
    df[f'prompt_{p_type}'] = df.apply(get_prompt_v1, args=(p_type, pipe.tokenizer), axis=1)

    # drop too long prompts
    df['tokens'] = df[f'prompt_{p_type}'].apply(num_tokens, args=(pipe.tokenizer,))
    df = df[df['tokens'] < args.max_length - args.min_new_tokens]
    print('after removing too long', df.shape)

    # generate text
    # df[f'gen_{p_type}'] = call_pipe_with_profiling(pipe=pipe, queries=df[f'prompt_{p_type}'].tolist(), batch_size=args.batch_size)
    df[f'gen_{p_type}'] = call_pipe(pipe=pipe, queries=df[f'prompt_{p_type}'].tolist(), batch_size=args.batch_size)

    # redo missing or too short
    df['tokens'] = df[f'gen_{p_type}'].apply(num_tokens, args=(pipe.tokenizer,))
    missing = df[(df[f'gen_{p_type}'].isna()) | (df['tokens'] < args.min_new_tokens)][f'prompt_{p_type}']
    if missing.any():
        print(f'filling {len(missing)} missing values for {p_type}')
        generated = [pipe(i)[0]['generated_text'] for i in tqdm(missing.values, desc='fill missing')]
        df.loc[missing.index, f'gen_{p_type}'] = generated

    df['tokens'] = df[f'prompt_{p_type}'].apply(num_tokens, args=(pipe.tokenizer,))
    df = df[df['tokens'] > 10].drop(columns='tokens')
    print(f'done generating {p_type} prompts for {len(df)} items')
    return df


@timeit
def get_data_new(args, tokenizer):
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = []
    for seed in range(5):
        data.append(pd.read_table(f'{args.folder}/reshuffle_{seed}/train.tsv'))
        data.append(pd.read_table(f'{args.folder}/reshuffle_{seed}/test.tsv'))
        data.append(pd.read_table(f'{args.folder}/reshuffle_{seed}/valid.tsv'))

    data = pd.concat(data).drop_duplicates(subset=['asin', 'user_id'])
    # vc = data.asin.value_counts()
    # data = data[data.asin.isin(vc[vc < 5].index)]
    items = data.asin.unique()
    # items = random.sample(list(data.asin.unique()), 100)[:10]

    kg = pd.read_table(
        f'{args.folder}/kg_readable_w_gen_desc_v2.tsv',
        index_col=0,
        usecols=['asin', 'title', 'description'],
    )
    kg = kg[kg.index.isin(items)]

    reviews = pd.read_table(f'{args.folder}/reviews_text.tsv')
    reviews = reviews[reviews.asin.isin(items)].dropna()
    reviews.review = reviews.review.apply(
        lambda x: tokenizer.convert_tokens_to_string(tokenizer.tokenize(x)[: args.max_review_length])
    )

    return kg, reviews


def add_reviews(df, p_type, pipe, args):
    # generate prompts

    reviews = pd.read_table(f'{args.folder}/reviews_text.tsv')
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
    print('after removing too long', df.shape)

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
    print(f'filling {len(missing)} missing values for {p_type}')
    generated = [pipe(i)[0]['generated_text'] for i in tqdm(missing.values)]
    df.loc[missing.index, f'gen_{p_type}_w_reviews'] = generated

    df['tokens'] = df[f'prompt_{p_type}_w_reviews'].apply(num_tokens, args=(pipe.tokenizer,))
    print(df.shape)
    df = df[df['tokens'] > args.min_new_tokens].drop(columns='tokens')
    print(f'done generating {p_type} prompts, {len(df)} items left out of initial {num_items}')
    return df


def main():
    args = parse_args()
    pipe = get_pipe(args, quantization='bfloat16')
    meta = get_data(args, pipe.tokenizer)
    print(f'generate text for {len(meta)} items')

    if args.no_reviews:
        for prompt_type in prompts:
            meta = process_type(meta, prompt_type, pipe, args)
            save(meta, f'{args.folder}/kg_readable_v4_proper_{args.model_name.split("/")[-1]}.tsv')

    if args.reviews:
        for prompt_type in prompts:
            meta = add_reviews(meta, prompt_type, pipe, args)
            save(meta, f'{args.folder}/kg_readable_v4_proper_w_reviews_{args.model_name.split("/")[-1]}.tsv')


# def drop_existing(args, kg, reviews):
#     existing_kg = pd.read_table(f'{args.folder}/kg_readable_w_gen_desc_v3_w_reviews.tsv', index_col=0)
#     existing_reviews = pd.read_table(f'{args.folder}/reviews_text.tsv')
#     kg = kg[~kg.index.isin(existing_kg.index)]
#     reviews = reviews[~reviews.asin.isin(existing_reviews.asin)]
#     return kg, reviews


# def new_main():
#     args = parse_args()
#     pipe = get_pipe(args)
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
#     kg, reviews = get_data_new(args, tokenizer)

#     # kg, reviews = drop_existing(args, kg, reviews)

#     for prompt_type in prompts:
#         kg = add_reviews(kg, reviews, prompt_type, pipe, args)
#         save(kg, f'{args.folder}/kg_readable_w_gen_desc_v3_w_reviews_missing.tsv')


if __name__ == '__main__':
    main()
    # new_main()
