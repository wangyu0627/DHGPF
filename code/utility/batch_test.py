import torch
import numpy as np
import multiprocessing
from utility.dataloader import Data
import utility.metrics
import random
import dgl

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('Inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def mini_batch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 1024)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def Test(dataset: Data, model, device, topK, flag_multicore, test_batch_size, long_tail=False):
    model = model.eval()
    if flag_multicore == 1:
        multicore = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(multicore)

    # top-20, 40, ..., 100
    model_results = {'precision': np.zeros(len(topK)),
                     'recall': np.zeros(len(topK)),
                     'HR': np.zeros(len(topK)),
                     'ndcg': np.zeros(len(topK))}
    with torch.no_grad():
        users = list(dataset.test_dict.keys())  # get user list to test
        # if test_batch_size > len(users) // 10:
        #     print(f"\tTest batch size is too big for dataset, please try a small one {len(users) // 10}")
        users_list, rating_list, ground_true_list = [], [], []
        num_batch = len(users) // test_batch_size + 1
        # item_batch = range(dataset.num_items)
        # num_batch = 1
        long_tail_rate = 0.
        for batch_users in mini_batch(users, batch_size=test_batch_size):
            exclude_users, exclude_items = [], []
            all_positive = dataset.get_user_pos_items(batch_users)
            ground_true = [dataset.test_dict[u] for u in batch_users]

            # batch_auxiliary, auxiliary_score = dataset.get_user_simi_users(batch_users)
            # item_batch_device = torch.Tensor(item_batch).long().to(device)
            batch_users_device = torch.Tensor(batch_users).long()#.to(device)
            #             batch_auxiliary_device = torch.Tensor(batch_auxiliary).long().to(device)
            #             auxiliary_score_device = torch.FloatTensor(auxiliary_score).to(device)
            # u_g_embeddings, pos_i_g_embeddings = SGL(batch_users_device, item_batch_device, [], feature_dict, mode='test')
            rating = model.getUsersRating(batch_users_device).detach().cpu()
            # rating = SGL.get_rating_for_test(batch_users_device, item_batch_device)

            # Positive items are excluded from the recommended list
            for i, items in enumerate(all_positive):
                exclude_users.extend([i] * len(items))
                exclude_items.extend(items)
            rating[exclude_users, exclude_items] = -1

            # get the top-K recommended list for all users
            _, rating_k = torch.topk(rating, k=max(topK))
            if long_tail:
                from time import time
                start_time = time()
                u_long_tail = 0
                for u_item in rating_k:
                    for item in dataset.long_tail_item:
                        if item in u_item[:20]:
                            u_long_tail += 1
                batch_long_tail_rate = float(u_long_tail)/float(test_batch_size * 20)
                long_tail_rate += batch_long_tail_rate
                print(time()-start_time)
            rating = rating.cpu()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            ground_true_list.append(ground_true)

        assert num_batch == len(users_list)
        enum_list = zip(rating_list, ground_true_list)

        if flag_multicore == 1:
            results = pool.map(test_one_batch, enum_list)
        else:
            results = []
            for single_list in enum_list:
                results.append(test_one_batch(single_list))
        for result in results:
            model_results['recall'] += result['recall']
            model_results['precision'] += result['precision']
            model_results['ndcg'] += result['ndcg']
            model_results['HR'] += result['HR']

        model_results['recall'] /= float(len(users))
        model_results['precision'] /= float(len(users))
        model_results['ndcg'] /= float(len(users))
        model_results['HR'] /= float(len(users))
        if flag_multicore == 1:
            pool.close()
        if long_tail:
            long_tail_rate = long_tail_rate / float(num_batch)
            return long_tail_rate, model_results
        return model_results


def test_one_batch(X):
    recommender_items = X[0].numpy()
    ground_true_items = X[1]
    r = utility.metrics.get_label(ground_true_items, recommender_items)
    precision, recall, ndcg, HR = [], [], [], []
    top_K = [20]
    for k_size in top_K:
        recall.append(utility.metrics.recall_at_k(r, k_size, ground_true_items))
        precision.append(utility.metrics.precision_at_k(r, k_size, ground_true_items))
        ndcg.append(utility.metrics.ndcg_at_k(r, k_size, ground_true_items))
        HR.append(utility.metrics.HR(recommender_items, k_size, ground_true_items))
    return {'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg), 'HR': np.array(HR)}


def Test_sparsity(dataset: Data, model, device, topK, test_batch_size):
    sparsity_results = []
    model = model.eval()
    # top-20, 40, ..., 100

    with torch.no_grad():
        for users in dataset.split_test_dict:
            model_results = {
                'precision': np.zeros(len(topK)),
                'recall': np.zeros(len(topK)),
                'hit': np.zeros(len(topK)),
                'ndcg': np.zeros(len(topK))
            }
            users_list, rating_list, ground_true_list = [], [], []
            num_batch = len(users) // test_batch_size + 1

            for batch_users in mini_batch(users, batch_size=test_batch_size):
                exclude_users, exclude_items = [], []
                all_positive = dataset.get_user_pos_items(batch_users)
                ground_true = [dataset.test_dict[u] for u in batch_users]

                batch_users_device = torch.Tensor(batch_users).long().to(device)

                rating = model.getUsersRating(batch_users_device).detach().cpu()

                # Positive items are excluded from the recommended list
                for i, items in enumerate(all_positive):
                    exclude_users.extend([i] * len(items))
                    exclude_items.extend(items)

                rating[exclude_users, exclude_items] = -1

                # get the top-K recommended list for all users
                _, rating_k = torch.topk(rating, k=max(topK))

                rating = rating.cpu()
                del rating

                users_list.append(batch_users)
                rating_list.append(rating_k.cpu())
                ground_true_list.append(ground_true)

            assert num_batch == len(users_list)
            enum_list = zip(rating_list, ground_true_list)

            results = []
            for single_list in enum_list:
                results.append(test_one_batch(single_list))

            for result in results:
                model_results['recall'] += result['recall']
                model_results['precision'] += result['precision']
                model_results['ndcg'] += result['ndcg']


            model_results['recall'] /= float(len(users))
            model_results['precision'] /= float(len(users))
            model_results['ndcg'] /= float(len(users))

            sparsity_results.append(model_results)

    return sparsity_results