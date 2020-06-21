import Graph
import Envir_setting as env
import os
import sys
import matplotlib.pyplot as plt


def initialization(nodes_in_one_row, min_lambda, max_lambda, min_mu, max_mu, frame_size, frame_rep):
    path1 = './arrival_patterns'
    arrival_list = os.listdir(path1)

    path2 = './transmission_patterns'
    trans_list = os.listdir(path2)

    edges = 2 * int(nodes_in_one_row) * (int(nodes_in_one_row) - 1)
    with open(
            './arrival_patterns/' + f'Arrivals(edge {edges}, lambda from {min_lambda:.4f} to {max_lambda:.4f}, frame_size {frame_size}, frame_rep {frame_rep}).txt',
            'r') as f:
        arrivals = f.readlines()

    with open('./transmission_patterns/' + f'Capacities({edges}, capacity from {min_mu:.2f} to {max_mu:.2f}).txt',
              'r') as f:
        capacities = f.read()
    return arrivals, capacities


def set_environment(nodes_in_one_row, min_lambda, max_lambda, min_mu, max_mu, frame_size, frame_rep):
    arr_patterns, tr_patterns = initialization(nodes_in_one_row, min_lambda, max_lambda, min_mu, max_mu, frame_size,
                                               frame_rep)
    tr_patterns = list(map(float, tr_patterns.split()))

    graph = Graph.GridGraph(nodes_in_one_row, arr_patterns, tr_patterns)
    return graph


def A_k_algorithm(p_seed, k, graph, estimator, frame_size, frame_rep):
    graph.init_estimator(estimator)

    for _ in range(frame_rep):
        graph.queue_snapshot()
        for t in range(1, frame_size + 1):
            matching = graph.A_k_matching(k=k, p_seed=p_seed, estimator=estimator, time=t)
            if not Graph.matching_checker(matching):
                raise Exception('Not matching ')
            graph.update_after_schedule(matching=matching, estimator=estimator)

            graph.queue_departure()
            graph.queue_arrival()
    sum_queues = sum([graph.graph.edges[edge]['backlog'] for edge in graph.graph.edges()])
    return sum_queues


def GMM_algorithm(graph, estimator, frame_size, frame_rep):
    graph.init_estimator(estimator)
    for _ in range(frame_rep):
        graph.queue_snapshot()
        for t in range(1, frame_size + 1):
            matching = graph.greedy_maximum_matching(estimator, t)
            if not Graph.matching_checker(matching):
                raise Exception('Not matching ')
            graph.update_after_schedule(matching=matching, estimator=estimator)

            graph.queue_departure()
            graph.queue_arrival()
    sum_queues = sum([graph.graph.edges[edge]['backlog'] for edge in graph.graph.edges()])
    return sum_queues


queue_trace_A_k = []
queue_trace_GMM = []
for i in range(1, 10):
    queue_trace_A_k.append(
        A_k_algorithm(0.2, 3, set_environment(4, 0.04 * i / 10, 0.12 * i / 10, 0.25, 0.75, 5000, 20), 'UCB', 5000,
                      20))
    queue_trace_GMM.append(
        GMM_algorithm(set_environment(4, 0.04 * i / 10, 0.12 * i / 10, 0.25, 0.75, 5000, 20), 'UCB', 5000,
                      20)
    )
plt.plot(queue_trace_A_k, 'r--', label='A^3-UCB')
plt.plot(queue_trace_GMM, 'b--', label='GMM')
plt.xlabel(r'\lambda', fontsize=15)
plt.ylabel('Queue', fontsize=15)
plt.legend()
plt.show()
