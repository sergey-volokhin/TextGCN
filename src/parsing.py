import argparse
import os
import sys
import time

import torch

from .utils import get_logger


def parse_args(s=None):
    kg_features_choices = ['base_desc', 'usecases', 'expert', 'description']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        required=True,
                        choices=[
                            'LightGCNScore', 'LightGCNRank',
                            'LTRLinearRank', 'LTRLinearWPop',
                            'LTRLinearScore', 'LTRLinearScoreWPop',
                            # 'adv_sampling',  # LightGCN with dynamic negative sampling
                        ],
                        help='which model to use')
    parser.add_argument('--data', '-d',
                        required=True,
                        type=str,
                        help='folder with the train/test data')
    parser.add_argument('--epochs', '-e',
                        default=1000,
                        type=int,
                        help='number of epochs to train')
    parser.add_argument('--evaluate_every', '--eval_every',
                        default=25,
                        type=int,
                        help='how often evaluation is performed during training (default: every 25 epochs)')
    parser.add_argument('--emb_size',
                        default=64,
                        type=int,
                        help='GNN embedding size')
    parser.add_argument('--batch_size',
                        default=2048,
                        type=int,
                        help='batch size for training and prediction')
    parser.add_argument('--uid',
                        type=str,
                        help="optional name for the model instead of generated uid")
    parser.add_argument('--no_save',
                        action='store_false',
                        dest='save',
                        help='whether to save the model (yes by default, no when specified --no_save)')
    parser.add_argument('--load',
                        type=str,
                        help='path to the model to load (continue training or predict)')
    parser.add_argument('--load_base',
                        type=str,
                        help='path to the base model to load for training the textual layer on top (LTR models)')
    parser.add_argument('--patience',
                        default=5,
                        type=int,
                        help='number of epochs without improvement before stopping')
    parser.add_argument('--no_train',
                        action='store_true',
                        help='whether to train')
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
                        default='debug',
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
    parser.add_argument('--slurm',
                        action='store_true',
                        help='whether using slurm to run (less output written in stdout)')

    ''' hyperparameters '''
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--reg_lambda',
                        default=0.001,
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

    score_params = parser.add_argument_group('scoring objective hyperparams')
    score_params.add_argument('--classification',
                              action='store_true',
                              help='whether to round the score predictions to the nearest integer')

    rank_params = parser.add_argument_group('ranking objective hyperparams')
    rank_params.add_argument('-k',
                             default=[20, 40],
                             type=int,
                             nargs='*',
                             help='list of k-s for metrics @k')
    rank_params.add_argument('--neg_samples',
                             default=1,
                             type=int,
                             help='number of negative examples to sample together with one positive')

    text_params = parser.add_argument_group('text model hyperparams')
    text_params.add_argument('--ltr_layers',
                             type=int,
                             nargs='*',
                             default=[],
                             help='additional hidden layers w sizes on top of GNN embeddings')
    text_params.add_argument('--freeze',
                             action='store_true',
                             help='whether to freeze GNN embeddings when learning linear model on top')
    text_params.add_argument('--emb_batch_size',
                             default=2048,
                             type=int,
                             help='batch size for calculating embeddings for textual data')
    text_params.add_argument('--encoder',
                             default='all-MiniLM-L6-v2',
                             type=str,
                             help='which encoder from SentenceTransformers is used to embed text. for example: '
                                  'google/bert_uncased_L-2_H-128_A-2 '
                                  'all-MiniLM-L6-v2 '
                                  'microsoft/deberta-v3-base '
                                  'microsoft/deberta-v3-xsmall '
                                  'roberta-large '
                                  'text-embedding-3-large')
    text_params.add_argument('--ltr_text_features',
                             type=str,
                             nargs='*',
                             choices=[f'{u}-{i}' for u in ['reviews'] + kg_features_choices for i in ['reviews'] + kg_features_choices],
                             default=['reviews-reviews', 'base_desc-base_desc', 'reviews-base_desc', 'base_desc-reviews'],
                             help='which textual features to use in the linear layer of LTR models in addition to LightGCN score')

    args = parser.parse_args(s.split()) if s is not None else parser.parse_args()
    return process_args(args)


def process_args(args):
    args.k = sorted(args.k)
    sys.setrecursionlimit(15000)  # this fixes tqdm bug
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else "cpu")
    if args.evaluate_every > args.epochs:
        args.logger.warn(
            f'Supplied args.evaluate_every ({args.evaluate_every}) '
            f'is greater than args.epochs ({args.epochs}). '
            'Setting args.evaluate_every to be args.epochs.',
        )
        args.evaluate_every = args.epochs
    args.kg_features = list({i.split('-')[1] for i in args.ltr_text_features} - {'reviews'})

    ''' paths '''
    args.data = os.path.join(args.data, '')  # make sure path ends with '/'
    if args.uid is None:
        args.uid = time.strftime("%m-%d-%Hh%Mm%Ss")
    args.save_path = os.path.join(
        'runs',
        # os.path.basename(os.path.dirname(os.path.dirname(args.data))),
        os.path.basename(args.data),
        args.model,
        args.uid,
    )
    os.makedirs(args.save_path, exist_ok=True)
    args.logger = get_logger(args)
    assert args.load is None or args.load_base is None, 'cannot both load base and load trained model'
    if args.load is not None and os.path.isdir(args.load):
        if os.path.exists(os.path.join(args.load, 'best.pkl')):
            args.load = os.path.join(args.load, 'best.pkl')
        elif os.path.exists(os.path.join(args.load, 'latest_checkpoint.pkl')):
            args.load = os.path.join(args.load, 'latest_checkpoint.pkl')
        else:
            args.logger.warn(f'No model to load found in {args.load}. Continuing without loading.')
            args.load = None

    return args
