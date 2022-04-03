import argparse
import os
import time

import torch

from utils import get_logger


def parse_args(s=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='lgcn',
                        choices=['lgcn',
                                 'lgcn_single',
                                 'lgcn_weights',
                                 'lgcn_single_attn',
                                 'lgcn_attn',
                                 'kg',
                                 'reviews',
                                 'ngcf'],
                        help='which model to use')
    parser.add_argument('--data',
                        default='data/amazon-book-2018/',
                        type=str,
                        help='folder with the train/test data')
    parser.add_argument('--epochs', '-e',
                        default=1000,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--emb_size',
                        default=64,
                        type=int,
                        help='embedding size')
    parser.add_argument('--batch_size',
                        default=2048,
                        type=int,
                        help='batch size for training and prediction')
    parser.add_argument('--uid',
                        default=False,
                        type=str,
                        help="optional name for the model instead of generated uid")

    parser.add_argument('--evaluate_every',
                        default=10,
                        type=int,
                        help='how often evaluation is performed during training (default: every 100 epochs)')
    parser.add_argument('-k',
                        default=[20, 40],
                        type=int,
                        nargs='*',
                        help='list of k-s for metrics @k')
    parser.add_argument('--save_model',
                        action='store_true',
                        help='whether to save the model')
    parser.add_argument('--load',
                        default=False,
                        type=str,
                        help='path to the model we want to load and continue training or evaluate')
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
    parser.add_argument('--keep_prob',
                        default=0.6,  # 0.6 for lightgcn, 0.1 for kgat
                        type=float,
                        help="probability of keeping the link in graph when doing dropout")
    parser.add_argument('--n_layers',
                        default=3,
                        type=int,
                        help="num layers")

    # text_hyper = parser.add_argument_group('text model hyperparams')
    # text_hyper.add_argument('--emb_batch_size',
    #                         default=256,
    #                         type=int,
    #                         help='batch size for embedding textual data')
    # text_hyper.add_argument('--bert-model',
    #                         default='google/bert_uncased_L-2_H-128_A-2',
    #                         # default='microsoft/deberta-v3-base',
    #                         # default='microsoft/deberta-v3-xsmall',
    #                         type=str,
    #                         help='version of BERT to use')

    # text_hyper.add_argument('--single_vector',
    #                         action='store_true',
    #                         help='whether to use one vector for all users or one per each')
    # text_hyper.add_argument('--layer_sizes',
    #                         default=[64, 32, 16],
    #                         nargs='*',
    #                         type=int,
    #                         help='Output sizes of every layer')
    # text_hyper.add_argument('--separator', '-sep',
    #                         default='[SEP]',
    #                         type=str,
    #                         dest='sep',
    #                         help='Separator for table comprehension')
    # text_hyper.add_argument('--freeze',
    #                         action='store_true',
    #                         help='whether to freeze textual item embeddings')

    args = parser.parse_args(s) if s is not None else parser.parse_args()

    ''' paths '''
    if args.load:
        args.save_path = os.path.dirname(args.load)
        args.uid = os.path.basename(args.save_path)
    else:
        if not args.uid:
            args.uid = time.strftime("%m-%d-%Hh%Mm%Ss")
        args.save_path = f'centralized_runs/{os.path.basename(os.path.dirname(args.data))}/{args.uid}'
        os.makedirs(args.save_path, exist_ok=True)
    args.data = os.path.join(args.data, '')  # make sure path ends with '/'

    ''' cuda '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else "cpu")

    args.k = sorted(args.k)
    args.logger = get_logger(args)

    return args
