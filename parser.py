import argparse
import os
import sys
import time

import torch

from utils import get_logger


def parse_args(s=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='lgcn',
                        choices=['lgcn',
                                 'gat',
                                 'gatv2',
                                 'gcn',
                                 'graphsage',
                                 'kg',
                                 'reviews',
                                 'reviews_loss',
                                 ],
                        help='which model to use')
    parser.add_argument('--aggr', '--aggregator',
                        default='mean',
                        choices=['mean', 'max', 'add'])
    parser.add_argument('--data',
                        default='data/subsampled/',
                        type=str,
                        help='folder with the train/test data')
    parser.add_argument('--epochs', '-e',
                        default=500,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--emb_size',
                        default=64,
                        type=int,
                        help='embedding size')
    parser.add_argument('--neg_samples',
                        default=1,
                        type=int,
                        help='number of negative examples to sample together with one positive')
    parser.add_argument('--batch_size',
                        default=2048,
                        type=int,
                        help='batch size for training and prediction')
    parser.add_argument('--uid',
                        default=False,
                        type=str,
                        help="optional name for the model instead of generated uid")
    parser.add_argument('--max_neighbors',
                        default=25,
                        type=int,
                        help='max number of neighbors to use when doing GraphSagePool')

    parser.add_argument('--evaluate_every',
                        default=25,
                        type=int,
                        help='how often evaluation is performed during training (default: every 100 epochs)')
    parser.add_argument('-k',
                        default=[20, 40],
                        type=int,
                        nargs='*',
                        help='list of k-s for metrics @k')
    parser.add_argument('--save',
                        action='store_true',
                        help='whether to save the model')
    parser.add_argument('--load',
                        default=False,
                        type=str,
                        help='path to the model to load')
    parser.add_argument('--predict',
                        action='store_true',
                        help='whether to save the predictions for test set')
    parser.add_argument('--gpu',
                        default='0',
                        type=str,
                        help='comma delimited list of GPUs that torch can see')
    parser.add_argument('--quiet', '-q',
                        action='store_true',
                        help='supress textual output in terminal (equivalent to error logging level)')
    parser.add_argument('--logging_level',
                        default='info',
                        type=str,
                        choices=['debug', 'info', 'warn', 'error'],
                        help='logging level')
    parser.add_argument('--seed',
                        default=1234,
                        type=int,
                        help='the random seed')

    parser.add_argument('--slurm',
                        action='store_true',
                        help='whether using slurm to run (less tqdm)')

    ''' hyperparameters '''
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--reg_lambda',
                        default=1e-4,
                        type=float,
                        help='the weight decay for l2 normalizaton')
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
    parser.add_argument('--ngcf',
                        action='store_true',
                        help="whether to use NGCF attention formula or original LightGCN")

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
    text_hyper.add_argument('--separator', '--sep',
                            default='[SEP]',
                            type=str,
                            dest='sep',
                            help='separator for table comprehension')

    args = parser.parse_args(s) if s is not None else parser.parse_args()

    ''' delete unused arguments if the model isn't textual '''
    if args.model not in ['kg', 'reviews', 'reviews_loss']:
        del args.emb_batch_size
        del args.sep
        del args.bert_model

    assert not (args.data in ['data/amazon-book/', 'data/amazon-book'] and args.emb_size != 64)

    ''' paths '''
    args.data = os.path.join(args.data, '')  # make sure path ends with '/'
    if args.load:
        args.save_path = os.path.dirname(args.load)
        args.uid = os.path.basename(args.save_path)
    else:
        if not args.uid:
            args.uid = time.strftime("%m-%d-%Hh%Mm%Ss")
        args.save_path = f'runs/{os.path.basename(os.path.dirname(args.data))}/{args.uid}'
        os.makedirs(args.save_path, exist_ok=True)

    ''' cuda '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else "cpu")
    # args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu else "cpu")

    args.k = sorted(args.k)
    args.logger = get_logger(args)

    sys.setrecursionlimit(15000)  # this fixes tqdm bug

    return args
