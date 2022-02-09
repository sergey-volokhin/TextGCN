import argparse
import os

import uuid
import torch

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce KGAT using DGL')
    # Data parameters
    parser.add_argument('--datapath',
                        type=str,
                        default='data/',
                        help='folder with the train/test data')
    parser.add_argument('--epochs', '-e',
                        type=int,
                        default=1000,
                        help='number of epochs')
    parser.add_argument('--embed_size',
                        type=int,
                        default=768,
                        help='embedding size')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='batch size')
    parser.add_argument('--embed_batch_size',
                        type=int,
                        default=256,
                        help='batch size for embedding')

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
    parser.add_argument('--bert-model',
                        type=str,
                        default='microsoft/deberta-v3-base',
                        help='version of BERT to use')
    parser.add_argument('--gpu',
                        type=str,
                        default='',
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
    parser.add_argument('--single_vector',
                        action='store_true',
                        help='whether to use one vector for all users or one per each')

    ''' these hyperparameters were in the original code, might be worth optimizing them for our data '''
    original_hyperparams = parser.add_argument_group('original hyperparameters')
    original_hyperparams.add_argument('--layer_size',
                                      nargs='*',
                                      type=int,
                                      default=[64, 32, 16],
                                      help='Output sizes of every layer')
    original_hyperparams.add_argument('--reg_lambda',
                                      type=float,
                                      default=1e-5,
                                      help='Regularization parameter')
    original_hyperparams.add_argument('--lr',
                                      type=float,
                                      default=0.0001,
                                      help='Learning rate.')
    original_hyperparams.add_argument('--mess_dropout',
                                      type=float,
                                      default=0.1,
                                      help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    original_hyperparams.add_argument('--separator', '-sep',
                                      type=str,
                                      default='[SEP]',
                                      dest='sep',
                                      help='Separator for table comprehension')

    args = parser.parse_args()

    ''' paths '''
    if args.load_path:
        args.save_path = os.path.dirname(args.load_path)
        args.uid = os.path.basename(args.save_path)
    else:
        args.uid = uuid.uuid4()
        args.save_path = f'results/{args.uid}'
        os.makedirs(args.save_path, exist_ok=True)

    args.k = sorted(args.k)
    args.layer_size = [args.embed_size] + args.layer_size
    args.logger = get_logger(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if args.gpu else torch.device('cpu')

    return args
