import Graph
import networkx as nx
import random
nodes = 3
edges = 2 * nodes * (nodes-1)
capacities = [round(random.random(),2) for _ in range(edges)]

G = Graph.Gridgraph(nodes, capacities)
MWM, sum_reward1 = G.max_weight_matching(with_queue=False)
# original_MWM =nx.max_weight_matching(G.G, weight='capacity')
# GMM, sum_reward2 = G.greedy_maximum_matching(with_queue=False)
G.draw(MWM,original_MWM)
