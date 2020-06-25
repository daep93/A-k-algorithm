import Graph
import Envir_setting as env
import os
import sys
import matplotlib.pyplot as plt
from tqdm import notebook

def initialization(edges, min_lambda, max_lambda, min_mu, max_mu, frame_size, frame_rep):
    path1 = './arrival_patterns'
    arrival_list = os.listdir(path1)

    path2 = './transmission_patterns'
    trans_list = os.listdir(path2)

    with open(
            './arrival_patterns/' + f'Arrivals(edge {edges}, lambda from {min_lambda:.4f} to {max_lambda:.4f}, frame_size {frame_size}, frame_rep {frame_rep}).txt',
            'r') as f:
        arrivals = f.readlines()

    with open('./transmission_patterns/' + f'Capacities({edges}, capacity from {min_mu:.2f} to {max_mu:.2f}).txt',
              'r') as f:
        capacities = f.read()
    return arrivals, capacities


def set_environment(nodes,edges, min_lambda, max_lambda, min_mu, max_mu, frame_size, frame_rep):
    arr_patterns, tr_patterns = initialization(edges, min_lambda, max_lambda, min_mu, max_mu, frame_size,frame_rep)
    tr_patterns = list(map(float, tr_patterns.split()))
    graph = Graph.LargeGraph(nodes,edges, arr_patterns, tr_patterns)
    return graph


def A_k_algorithm(p_seed, k, graph, estimator, frame_size, frame_rep):
    graph.reinit()
    graph.init_estimator(estimator)

    for _ in notebook.tqdm(range(frame_rep)):
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

def dA_k_algorithm(p_seed, k, graph, estimator, frame_size, frame_rep):
    graph.reinit()
    graph.init_estimator(estimator)

    for _ in notebook.tqdm(range(frame_rep)):
        graph.queue_snapshot()
        for t in range(1, frame_size + 1):
            matching = graph.dA_k_matching(k=k, p_seed=p_seed, estimator=estimator, time=t)
            if not Graph.matching_checker(matching):
                raise Exception('Not matching ')
            graph.update_after_schedule(matching=matching, estimator=estimator)

            graph.queue_departure()
            graph.queue_arrival()
    sum_queues = sum([graph.graph.edges[edge]['backlog'] for edge in graph.graph.edges()])
    return sum_queues

def GMM_algorithm(graph, estimator, frame_size, frame_rep):
    graph.init_estimator(estimator)
    for _ in notebook.tqdm(range(frame_rep)):
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



