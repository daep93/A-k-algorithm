import Graph
import networkx as nx
import random
nodes = 3
edges = 2 * nodes * (nodes-1)
capacities = [round(random.random(), 2) for _ in range(edges)]

G = Graph.Gridgraph(nodes, capacities)
# MWM, sum_reward1 = G.max_weight_matching(with_queue=False)
# original_MWM =nx.max_weight_matching(G.G, weight='capacity')
# GMM, sum_reward2 = G.greedy_maximum_matching(with_queue=False)
for edge in G.G.edges:
    G.G.edges[edge]['scheduled'] = False
# G.draw([])
G.G.edges[((0, 0), (0, 1))]['scheduled'] = True
G.G.edges[((2, 1), (2, 2))]['scheduled'] = True
G.G.edges[((1, 0), (2, 0))]['scheduled'] = True
# for t in range(1,10):
Aug, sum_reward3 = G.augmentation(3, 0.4, False, 'TS', 1)
print('success' if G.matching_checker(Aug) else 'fail')
G.draw(Aug)
