import argparse
import os
import time

import torch

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='text',
                        choices=['lightgcn', 'text'],
                        help='which model to use')
    parser.add_argument('--datapath',
                        type=str,
                        default='data/amazon-book-2018/',
                        help='folder with the train/test data')
    parser.add_argument('--epochs', '-e',
                        type=int,
                        default=1000,
                        help='number of epochs')
    parser.add_argument('--emb_size',
                        type=int,
                        default=64,
                        help='embedding size')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='batch size for training and prediction')
    parser.add_argument('--uid',
                        type=str,
                        default=False,
                        help="optional name for the model instead of generated uid")

    parser.add_argument('--evaluate_every',
                        type=int,
                        default=100,
                        help='how often evaluation is performed during training (default: every 100 epochs)')
    parser.add_argument('-k',
                        type=int,
                        default=[20, 40],
                        nargs='*',
                        help='list of k-s for metrics @k')
    parser.add_argument('--save_model',
                        action='store_true',
                        help='whether to save the model')
    parser.add_argument('--load_path',
                        type=str,
                        default=False,
                        help='path to the model we want to load and continue training or evaluate')
    parser.add_argument('--predict',
                        action='store_true',
                        help='whether to save the predictions for test set')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='comma delimited list of GPUs that torch can see')
    parser.add_argument('--quiet', '-q',
                        action='store_true',
                        help='supress textual output in terminal (equivalent to error logging level)')
    parser.add_argument('--logging_level',
                        type=str,
                        default='info',
                        choices=['debug', 'info', 'warn', 'error'],
                        help='logging level')
    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='the random seed')

    ''' hyperparameters '''
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--reg_lambda',
                        type=float,
                        default=1e-5,
                        help='the weight decay for l2 normalizaton')

    text_hyper = parser.add_argument_group('text model hyperparams')
    text_hyper.add_argument('--emb_batch_size',
                            type=int,
                            default=256,
                            help='batch size for embedding')
    text_hyper.add_argument('--bert-model',
                            type=str,
                            default='microsoft/deberta-v3-base',
                            help='version of BERT to use')
    text_hyper.add_argument('--single_vector',
                            action='store_true',
                            help='whether to use one vector for all users or one per each')
    text_hyper.add_argument('--layer_size',
                            nargs='*',
                            type=int,
                            default=[64, 32, 16],
                            help='Output sizes of every layer')
    text_hyper.add_argument('--mess_dropout',
                            type=float,
                            default=0.1,
                            help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    text_hyper.add_argument('--separator', '-sep',
                            type=str,
                            default='[SEP]',
                            dest='sep',
                            help='Separator for table comprehension')
    text_hyper.add_argument('--freeze',
                            action='store_true',
                            help='whether to freeze textual item embeddings')

    lightgcn = parser.add_argument_group('lightGCN hyperparams')
    lightgcn.add_argument('--n_layers',
                          type=int,
                          default=3,
                          help="num layers")
    lightgcn.add_argument('--dropout',
                          action='store_true',
                          help="using the dropout or not")
    lightgcn.add_argument('--keep_prob',
                          type=float,
                          default=0.6,
                          help="dropout??")

    args = parser.parse_args()

    ''' paths '''
    if args.load_path:
        args.save_path = os.path.dirname(args.load_path)
        args.uid = os.path.basename(args.save_path)
    else:
        if not args.uid:
            args.uid = time.strftime("%m-%d-%Hh%Mm%Ss")
        args.save_path = f'runs/{os.path.basename(os.path.dirname(args.datapath))}/{args.uid}'
        os.makedirs(args.save_path, exist_ok=True)
    args.datapath = os.path.join(args.datapath, '')  # make sure path ends with '/'

    ''' cuda '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else "cpu")

    args.k = sorted(args.k)
    args.logger = get_logger(args)

    args.layer_size = [args.emb_size] + args.layer_size

    return args
