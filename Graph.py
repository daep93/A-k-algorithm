import networkx as nx
from tqdm import tqdm_notebook
import numpy as np
import math
from collections import namedtuple
import matplotlib.pyplot as plt


class Gridgraph:

    def __init__(self, _nodes, _capacities):
        _capacities=iter(_capacities)
        self.G = nx.grid_2d_graph(_nodes, _nodes)
        self.pos = nx.spring_layout(self.G)
        for i, j in self.G.edges:
            self.G[i][j]['capacity'] = next(_capacities)
            self.G[i][j]['backlog'] = 0
            self.G[i][j]['scheduled'] = False

    def neighbor_edges(self, edge):

        end1, end2 = edge
        neighbor_1 = [(end1, neighbor) for neighbor in self.G.adj[end1]]
        neighbor_1.remove((end1, end2))
        neighbor_2 = [(end2, neighbor) for neighbor in self.G.adj[end2]]
        neighbor_2.remove((end2, end1))

        neighbors = neighbor_1 + neighbor_2

        return neighbors

    def draw(self, *matchings):
        num = len(matchings)
        cnt = 1
        for matching in matchings:

            plt.subplot('1' + str(num)+str(cnt))
            cnt += 1

            nx.draw(self.G, self.pos, with_labels=True)
            # nx.draw_networkx_nodes(self.G, self.pos, nodelist=self.G.nodes, node_color='r', node_size=1000, alpha=0.5)
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=nx.get_edge_attributes(self.G, 'capacity'))
            nx.draw_networkx_edges(self.G, self.pos, edgelist=matching, width=8, alpha=0.5, edge_color='b')
        plt.show()

    def refresh(self):
        for i, j in self.G.edges:
            self.G[i][j]['change'] = False
            self.G[i][j]['instance rate'] = 0

        for _node in self.G.nodes:
            self.G.node[_node]['touched'] = [False, False]

    def max_weight_matching(self, with_queue):
        weight = 'capacity'

        if with_queue:
            for i, j in self.G.edges:
                self.G[i][j]['weighted capacity'] = self.G[i][j]['backlog'] * self.G[i][j]['capacity']
            weight = 'weighted capacity'

        matching = nx.max_weight_matching(self.G, weight=weight)

        sum_reward = 0
        for i, j in matching:
            sum_reward += self.G[i][j][weight]

        return matching, sum_reward

    def greedy_maximum_matching(self, with_queue):
        Edge = namedtuple('Edge', 'edge, weight')
        weight = 'capacity'

        if with_queue:

            for i, j in self.G.edges:
                self.G[i][j]['weighted capacity'] = self.G[i][j]['backlog'] * self.G[i][j]['capacity']
            weight = 'weighted capacity'

        edges = [Edge(edge=e, weight=self.G.edges[e][weight]) for e in self.G.edges]
        sorted_edges = [e.edge for e in sorted(edges, key=lambda l: l.weight, reverse=True)]

        matching = []
        print('sorted edges: ', sorted_edges)
        while len(sorted_edges) != 0:

            next_edge = sorted_edges[0]
            print('next edge: ', next_edge)

            matching.append(next_edge)
            sorted_edges.remove(next_edge)

            edge_adj_list = self.neighbor_edges(next_edge)
            print('edge_adj_list: ', edge_adj_list)
            print('remain edges: ', sorted_edges)

            for target in edge_adj_list:
                reversed_target = tuple(reversed(target))

                if target in sorted_edges:
                    print('delete', target)
                    sorted_edges.remove(target)
                if reversed_target in sorted_edges:
                    print('delete', reversed_target)
                    sorted_edges.remove(reversed_target)
        sum_reward = 0
        for e in matching:
            sum_reward += self.G.edges[e][weight]

        return matching, sum_reward
