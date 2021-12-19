import heapq
import multiprocessing
from collections import defaultdict
from itertools import repeat

import numpy as np
import torch
from tqdm import tqdm


def l2_loss_mean(x):
    ''' mean( sum(t ** 2) / 2) '''
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1
    return 0


def recall_at_k(hit, k, all_pos_num):
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    return np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def test_one_user(rating, training_items, user_pos_test, n_items, ks):

    all_items = set(range(n_items))
    test_items = list(all_items - set(training_items))
    item_score = {i: rating[i] for i in test_items}
    k_max_item_score = heapq.nlargest(max(ks), item_score, key=item_score.get)
    user_pos_test = set(user_pos_test)
    rankedlist = [int(i in user_pos_test) for i in k_max_item_score]

    return {
        'recall': np.array([recall_at_k(rankedlist, K, len(user_pos_test)) for K in ks]),
        'precision': np.array([precision_at_k(rankedlist, K) for K in ks]),
        'hit_ratio': np.array([hit_at_k(rankedlist, K) for K in ks]),
        'ndcg': np.array([ndcg_at_k(rankedlist, K) for K in ks]),
    }


def calculate_metrics_cpu(embedding, train_user_dict, test_user_dict, all_item_id_range, batch_size, ks):
    result = defaultdict(lambda: np.zeros(len(ks)))
    user_ids_batches = [list(test_user_dict)[i: i + batch_size] for i in range(0, len(test_user_dict), batch_size)]
    n_users = len(test_user_dict)
    cores = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(cores)
    with torch.no_grad():
        for user_ids_batch in tqdm(user_ids_batches, dynamic_ncols=True, leave=False, desc='evaluating'):
            cf_scores_batch = torch.matmul(embedding[user_ids_batch], embedding[all_item_id_range].transpose(0, 1))
            cf_scores_batch = cf_scores_batch.cpu()
            train_dict_batch = [train_user_dict[u] for u in user_ids_batch]
            test_dict_batch = [test_user_dict[u] for u in user_ids_batch]
            batch_result = pool.starmap(test_one_user, zip(cf_scores_batch, train_dict_batch, test_dict_batch, all_item_id_range, repeat(ks)))
            for re in batch_result:
                for i in re:
                    result[i] += re[i] / n_users
    pool.close()
    return result


def calculate_metrics(embedding, train_user_dict, test_user_dict, all_item_id_range, ks):
    result = defaultdict(lambda: np.zeros(len(ks)))
    n_users = len(test_user_dict)
    with torch.no_grad():
        for u_id, pos_item_l in tqdm(test_user_dict.items(), dynamic_ncols=True, leave=False, desc='evaluating'):
            score = torch.matmul(embedding[u_id], embedding[all_item_id_range].transpose(0, 1))
            score[train_user_dict[u_id]] = 0.0
            _, rank_indices = torch.topk(score, k=max(ks), largest=True)
            rank_indices = rank_indices.cpu().numpy()
            rankedlist = [int(i in pos_item_l) for i in rank_indices]
            for ind, K in enumerate(ks):
                result['recall'][ind] += recall_at_k(rankedlist, K, len(pos_item_l)) / n_users
                result['precision'][ind] += precision_at_k(rankedlist, K) / n_users
                result['hit_ratio'][ind] += hit_at_k(rankedlist, K) / n_users
                result['ndcg'][ind] += ndcg_at_k(rankedlist, K) / n_users
    return result
