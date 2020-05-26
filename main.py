import Graph
import Envir_setting as env
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

sys.stdin = open('input.txt', 'r')


def initialization():
    path1 = './arrival_patterns'
    arrival_list = os.listdir(path1)

    path2 = './transmission_patterns'
    trans_list = os.listdir(path2)

    print(f'Arrival list: {arrival_list}')
    print(f'Transmission_list: {trans_list}')

    option = int(input('1. New environment, 2. Select existing pattern: '))

    seed, nodes = input('Input seed and the number of nodes in one row: ').split()
    lamb, time = input('Input lambda and time for arrival pattern: ').split()
    lower, upper = input('Input lower and upper bound: ').split()
    edges = 2 * int(nodes) * (int(nodes) - 1)

    if option == 1:
        arrival_pattern = env.generate_arrivals(rnd_seed=seed, edges=edges, lamb=lamb, time=time)
        trans_pattern = env.generate_transmissions(rnd_seed=seed, edges=edges, lower=lower, upper=upper)

    if option == 2:
        with open('./arrival_patterns/' + f'ar_seed({seed}, {edges}, {lamb}, {time}).txt', 'r') as f:
            arrival_pattern = f.read()

        with open('./transmission_patterns/' + f'tr_seed({seed}, {edges}, {lower}, {upper}).txt', 'r') as f:
            trans_pattern = f.read()

    return arrival_pattern, trans_pattern, int(nodes), int(time)


def schedule_with_queue(algorithm, estimator, frame_size, frame_rep, for_aug):
    G = Graph.Gridgraph(nodes, tr_pattern)
    regrets = [0]
    if algorithm == 'MWM':
        for _ in range(1, frame_rep + 1):
            G.estimaotr_init(estimator)

            for t in range(1, frame_size + 1):
                matching, sum_reward = G.max_weight_matching(with_queue=True, estimator=estimator, time=t)
                G.after_schedule_update(matching=matching, estimator=estimator)

            G.queue_update()
    elif algorithm == 'GMM':
        for _ in range(1, frame_rep + 1):
            _, max_reward = G.max_weight_matching(with_queue=True, estimator='capacity', time=0)

            G.estimaotr_init(estimator)
            for t in range(1, frame_size + 1):
                matching, sum_reward = G.greedy_maximum_matching(with_queue=True, estimator=estimator, time=t)
                G.after_schedule_update(matching=matching, estimator=estimator)
                regrets.append(regrets[-1] + max_reward - sum_reward)

            G.queue_update()
    elif algorithm == 'AUG':
        k, p_seed = for_aug
        _, max_reward = G.max_weight_matching(with_queue=True, estimator='capacity', time=0)

        G.estimaotr_init(estimator)
        for _ in range(frame_rep):
            for t in range(1, frame_size + 1):
                matching, sum_reward = G.augmentation(k=int(k), p_seed=float(p_seed), with_queue=True, estimator=estimator,
                                                      time=t)
                G.after_schedule_update(matching=matching, estimator=estimator)
                regrets.append(regrets[-1] + max_reward - sum_reward)

            G.queue_update()

    return G, regrets


def schedule_without_queue(algorithm, estimator, Total_time, for_aug):
    G = Graph.Gridgraph(nodes, tr_pattern)
    regrets = [0]
    if algorithm == 'MWM':
        for t in range(1, Total_time + 1):
            matching, sum_reward = G.max_weight_matching(with_queue=False, estimator=estimator, time=t)
            G.after_schedule_update(matching=matching, estimator=estimator)
    elif algorithm == 'GMM':
        _, max_reward = G.max_weight_matching(with_queue=False, estimator='capacity', time=0)
        for t in range(1, Total_time + 1):
            matching, sum_reward = G.greedy_maximum_matching(with_queue=False, estimator=estimator, time=t)
            G.after_schedule_update(matching=matching, estimator=estimator)
            regrets.append(regrets[-1] + max_reward - sum_reward)
    elif algorithm == 'AUG':
        k, p_seed = for_aug
        _, max_reward = G.max_weight_matching(with_queue=False, estimator='capacity', time=0)

        for t in range(1, Total_time + 1):
            matching, sum_reward = G.augmentation(k=int(k), p_seed=float(p_seed), with_queue=False, estimator=estimator,
                                                  time=t)
            if not Graph.matching_checker(matching):
                return None, regrets

            G.after_schedule_update(matching=matching, estimator=estimator)
            regrets.append(regrets[-1] + max_reward - sum_reward)

    return G, regrets


ar_pattern, tr_pattern, nodes, time = initialization()
tr_pattern = list(map(float, tr_pattern.split()))
algo, esti = input('algorithm, estimator, else: ').split()
if algo == 'AUG':
    aug_info = list(input('k, p_seed: ').split())

grid_graph, regrets = schedule_without_queue(algorithm=algo, estimator=esti, Total_time=time, for_aug=aug_info)
plt.plot(regrets, 'b--', label= algo+' '+esti)
plt.xlabel('Slot', fontsize=15)
plt.ylabel('Regret', fontsize=15)
plt.legend()
plt.show()
