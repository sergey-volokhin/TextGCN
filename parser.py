import argparse
import os
import sys
import time

import torch

from utils import get_logger


def parse_args(s=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        required=True,
                        choices=['lgcn',  # BaseModel, custom LightGCN
                                 'lightgcn', 'gat', 'gatv2', 'gcn', 'graphsage',  # torch_geometric
                                 'ltr_linear', 'ltr_linear_pop',
                                 'text', 'reviews', 'kg',
                                 'adv_sampling',  # dynamic negative sampling
                                 ],
                        help='which model to use')
    parser.add_argument('--aggr', '--aggregator',
                        choices=['mean', 'max', 'add'],
                        help='neighbor node aggregation function used in torch_geometric models')
    parser.add_argument('--data',
                        default='data/subsampled/',
                        type=str,
                        help='folder with the train/test data')
    parser.add_argument('--epochs', '-e',
                        default=1000,
                        type=int,
                        help='number of epochs to train')
    parser.add_argument('--emb_size',
                        default=64,
                        type=int,
                        help='GNN embedding size')
    parser.add_argument('--neg_samples',
                        default=1,
                        type=int,
                        help='number of negative examples to sample together with one positive')
    parser.add_argument('--batch_size',
                        default=2048,
                        type=int,
                        help='batch size for training and prediction')
    parser.add_argument('--uid',
                        type=str,
                        help="optional name for the model instead of generated uid")

    parser.add_argument('--evaluate_every', '--eval_every',
                        default=25,
                        type=int,
                        help='how often evaluation is performed during training (default: every 25 epochs)')
    parser.add_argument('-k',
                        default=[20, 40],
                        type=int,
                        nargs='*',
                        help='list of k-s for metrics @k')
    parser.add_argument('--save',
                        action='store_false',
                        help='whether to save the model (yes by default)')
    parser.add_argument('--load',
                        type=str,
                        help='path to the model to load')
    parser.add_argument('--predict',
                        action='store_true',
                        help='whether to save the predictions for test set')
    parser.add_argument('--gpu',
                        default='0',
                        type=str,
                        help="comma delimited list of GPUs that torch can see ('' if no GPU)")
    parser.add_argument('--quiet', '-q',
                        action='store_true',
                        help='supress textual output in terminal (equivalent to error logging level)')
    parser.add_argument('--logging_level',
                        default='info',
                        type=str,
                        choices=['debug', 'info', 'warn', 'error'],
                        help='logging level')
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help='random seed')
    parser.add_argument('--reshuffle',
                        action='store_true',
                        help='whether to reshuffle the train-test split or use the existing one')

    parser.add_argument('--freeze_embeddings',
                        action='store_true')

    parser.add_argument('--slurm',
                        action='store_true',
                        help='whether using slurm to run (less output written in stdout)')

    ''' hyperparameters '''
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--reg_lambda',
                        default=1e-4,
                        type=float,
                        help='the weight decay for L2 normalizaton')
    parser.add_argument('--dropout',
                        default=0.4,  # 0.4 for lightgcn, 0.9 for kgat
                        type=float,
                        help="dropout probability for links in graph")
    parser.add_argument('--n_layers',
                        default=3,
                        type=int,
                        help="num layers")
    parser.add_argument('--single',
                        action='store_true',
                        help="whether to use the 'single' verison of the model or not")

    text_hyper = parser.add_argument_group('text model hyperparams')
    text_hyper.add_argument('--emb_batch_size',
                            default=256,
                            type=int,
                            help='batch size for embedding textual data')
    text_hyper.add_argument('--bert_model',
                            default='all-MiniLM-L6-v2',
                            type=str,
                            choices=['google/bert_uncased_L-2_H-128_A-2',
                                     'all-MiniLM-L6-v2',
                                     'microsoft/deberta-v3-base',
                                     'microsoft/deberta-v3-xsmall',
                                     'roberta-large',
                                     ],
                            help='version of BERT to use')
    text_hyper.add_argument('--dist_fn',
                            default='euclid',
                            choices=['euclid', 'cosine_minus'],
                            help='distance metric used in textual loss')
    text_hyper.add_argument('--separator', '--sep',
                            default='[SEP]',
                            type=str,
                            dest='sep',
                            help='separator for table comprehension (KG model)')
    text_hyper.add_argument('--weight',
                            help='formula for semantic loss')
    text_hyper.add_argument('--pos',
                            default='avg',
                            choices=['user', 'avg', 'kg'],
                            help='how to represent the positive items from the sampled triplets')
    text_hyper.add_argument('--neg',
                            default='avg',
                            choices=['avg', 'kg'],
                            help='how to represent the negative items from the sampled triplets')
    args = parser.parse_args(s) if s is not None else parser.parse_args()

    asserts(args)

    ''' paths '''
    args.data = os.path.join(args.data, '')  # make sure path ends with '/'
    if args.load is not None:
        if args.uid is None:
            args.save_path = os.path.dirname(args.load)
            args.uid = os.path.basename(args.save_path)
        else:
            args.save_path = f'runs/no_scheduler/{os.path.basename(os.path.dirname(args.data))}/{args.uid}'
    else:
        if args.uid is None:
            args.uid = time.strftime("%m-%d-%Hh%Mm%Ss")
            # args.uid = f'{args.model}_{args.weight}_{args.dist_fn}'
        args.save_path = f'runs/no_scheduler/{os.path.basename(os.path.dirname(args.data))}/{args.uid}'
    os.makedirs(args.save_path, exist_ok=True)

    ''' cuda '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else "cpu")

    args.k = sorted(args.k)
    args.logger = get_logger(args)
    sys.setrecursionlimit(15000)  # this fixes tqdm bug

    return args


def asserts(args):
    if args.model in ['gat', 'gatv2', 'gcn', 'graphsage']:
        assert args.aggr is not None, 'set up the aggregator function for torch_geometric model'
    elif args.model in ['text', 'reviews', 'kg']:
        assert args.weight is not None, 'set the weight for model taht uses semantic loss'
    elif args.model in ['ltr_linear', 'ltr_simple', 'ltr_linear_pop']:
        assert args.load is not None, 'you need to load a pretrained LightGCN model'
    if 'ltr' in args.model:
        assert args.uid is not None, 'set uid for LTR model to avoid overwriting pretrained model'
