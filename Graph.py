import networkx as nx
# from tqdm import tqdm_notebook
import numpy as np
import math
from recordclass import recordclass
import matplotlib.pyplot as plt


class Gridgraph:

    def __init__(self, _nodes, _capacities):
        _capacities = iter(_capacities)
        self.G = nx.grid_2d_graph(_nodes, _nodes)
        self.pos = nx.spring_layout(self.G)
        for edge in self.G.edges:
            self.G.edges[edge]['capacity'] = next(_capacities)
            self.G.edges[edge]['backlog'] = 0
            self.G.edges[edge]['scheduled'] = False
            self.G.edges[edge]['sample mean'] = 1
            self.G.edges[edge]['alpha'] = 0
            self.G.edges[edge]['beta'] = 0

    def neighbor_edges(self, ty, obj, past):
        """
        func_name: neighbor_edges
            This function returns the neighbor edges of 'obj'.
            the left point of all neighbor edge is obj.
        :param ty:
            Neighbor edges of edge  -> input 'e'
            Neighbor edges of point -> input 'p'.
        :param obj:
            information of point or edge
        :param past:
            if you want not to consider 'past' point, then function will return the result
            except edges where end point is 'past'.
        :return:
            list of neighbor edges of 'obj'
        """

        if ty == 'e':
            end1, end2 = obj
            neighbor_1 = set([(end1, neighbor) for neighbor in self.G.adj[end1]])
            neighbor_1.remove((end1, end2))
            neighbor_2 = set([(end2, neighbor) for neighbor in self.G.adj[end2]])
            neighbor_2.remove((end2, end1))

            neighbors = list(neighbor_1.union(neighbor_2))
        else:
            adj = list(self.G.adj[obj])
            if past is not None:
                adj.remove(past)
            neighbors = [(obj, neighbor) for neighbor in adj]

        return neighbors

    def draw(self, *matchings):
        num = len(matchings)
        cnt = 1
        for matching in matchings:
            plt.subplot('1' + str(num) + str(cnt))
            cnt += 1

            nx.draw(self.G, self.pos, with_labels=True)
            # nx.draw_networkx_nodes(self.G, self.pos, nodelist=self.G.nodes, node_color='r', node_size=1000, alpha=0.5)
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=nx.get_edge_attributes(self.G, 'capacity'))
            nx.draw_networkx_edges(self.G, self.pos, edgelist=matching, width=8, alpha=0.5, edge_color='b')
        plt.show()

    def refresh(self):
        """
        Only for augmentation algorithm.
        Initialize 'change' variable and touched trace
        :return: None
        """
        for edge in self.G.edges:
            self.G.edges[edge]['change'] = False

        for _node in self.G.nodes:
            self.G.nodes[_node]['touched'] = False

    def estimator_update(self, estimator, with_queue, time):
        """
        func_name: estimator_update
            At each time, this function updates estimators
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param with_queue:
            want to consider queue -> True
            otherwise -> False
        :param time:
            time slot
        :return: None
        """
        if estimator == 'TS':
            if with_queue:
                for edge in self.G.edges:
                    self.G.edges[edge]['TS'] = self.G.edges[edge]['weight'] * np.random.beta(
                        self.G.edges[edge]['alpha'] + 1, self.G.edges[edge]['beta'] + 1
                    )
            else:
                for edge in self.G.edges:
                    self.G.edges[edge]['TS'] = np.random.beta(
                        self.G.edges[edge]['alpha'] + 1, self.G.edges[edge]['beta'] + 1
                    )

        elif estimator == 'UCB':
            if with_queue:
                for edge in self.G.edges:
                    self.G.edges[edge]['UCB'] = self.G.edges[edge]['weight'] * self.G.edges[edge]['sample mean'] \
                                                + math.sqrt(
                        (self.G.number_of_nodes() + 1) * math.log(time) / (
                                self.G.edges[edge]['alpha'] + self.G.edges[edge]['beta'] + 1)
                    )
            else:
                for edge in self.G.edges:
                    self.G.edges[edge]['UCB'] = self.G.edges[edge]['sample mean'] \
                                                + math.sqrt(
                        (self.G.number_of_nodes() + 1) * math.log(time) / (
                                self.G.edges[edge]['alpha'] + self.G.edges[edge]['beta'] + 1)
                    )

        else:
            return

    def max_weight_matching(self, with_queue, estimator, time):
        """
        func_name: max_weight_matching
            This function calculate max weight matching of given graph and return the matching
            and mean reward of the matching.

        :param with_queue:
            want to consider queue -> True
            otherwise -> False
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return:
            return the max weight matching and mean reward of the matching
        """
        weight = 'capacity'

        if with_queue:
            # For calculation of queue ratio
            backlogs_snapshot = set([self.G.edges[edge]['backlog'] for edge in self.G.edges])
            backlog_max = max(backlogs_snapshot)

            weight = 'weighted capacity'
            if estimator == 'capacity':
                estimator = weight

            if backlog_max == 0:
                backlog_max = 1

            for edge in self.G.edges:
                self.G.edges[edge]['weight'] = self.G.edges[edge]['backlog'] / backlog_max
                self.G.edges[edge][weight] = self.G.edges[edge]['weight'] * self.G.edges[edge]['capacity']

        self.estimator_update(estimator, with_queue, time)
        matching = nx.max_weight_matching(self.G, weight=estimator)

        sum_reward = 0
        for i, j in matching:
            sum_reward += self.G[i][j][weight]

        return matching, sum_reward

    def greedy_maximum_matching(self, with_queue, estimator, time):
        """
        func_name: greedy_maximum_matching
            This function calculate greedy maximum weight matching of given graph
            and return the matching and mean reward of the matching.

        :param with_queue:
            want to consider queue -> True
            otherwise -> False
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return:
            return the greedy maximum matching and mean reward of the matching
        """
        Edge = recordclass('Edge', 'edge, weight')
        weight = 'capacity'

        if with_queue:
            # For calculation of queue ratio
            backlogs_snapshot = set([self.G.edges[edge]['backlog'] for edge in self.G.edges])
            backlog_max = max(backlogs_snapshot)

            weight = 'weighted capacity'
            if estimator == 'capacity':
                estimator = weight

            if backlog_max == 0:
                backlog_max = 1

            for edge in self.G.edges:
                self.G.edges[edge]['weight'] = self.G.edges[edge]['backlog'] / backlog_max
                self.G.edges[edge][weight] = self.G.edges[edge]['weight'] * self.G.edges[edge]['capacity']

        edges = [Edge(edge=e, weight=self.G.edges[e][estimator]) for e in self.G.edges]
        sorted_edges = [e.edge for e in sorted(edges, key=lambda l: l.weight, reverse=True)]

        matching = []
        while len(sorted_edges) != 0:

            next_edge = sorted_edges[0]

            matching.append(next_edge)
            sorted_edges.remove(next_edge)

            edge_adj_list = self.neighbor_edges('e',next_edge,None)

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

    def augmentation(self, k, p_seed, with_queue, estimator, time):
        """
        func_name: augmentation
                    This function calculate a matching by augmenting
                    and return the matching and mean reward of the matching.
        :param k:
            intended size (= upper bound of augmentation size)
        :param p_seed:
            the probability of seed
        :param with_queue:
            want to consider queue -> True
            otherwise -> False
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return:
            return the calculated matching by augmenting and mean reward of the matching

        """
        weight = 'capacity'

        # Change the weight
        if with_queue:
            # For calculation of queue ratio
            backlogs_snapshot = set([self.G.edges[edge]['backlog'] for edge in self.G.edges])
            backlog_max = max(backlogs_snapshot)

            weight = 'weighted capacity'
            if estimator == 'capacity':
                estimator = weight

            if backlog_max == 0:
                backlog_max = 1

            for edge in self.G.edges:
                self.G.edges[edge]['weight'] = self.G.edges[edge]['backlog'] / backlog_max
                self.G.edges[edge][weight] = self.G.edges[edge]['weight'] * self.G.edges[edge]['capacity']

        # Update estimator and refresh scheduled
        self.estimator_update(estimator, with_queue, time)
        self.refresh()

        # Generate seeds for augmenting
        augmentations = dict()
        Aug = recordclass('Augmentation', 'seed, active_pnt, next_pnt, alive, cur_size, max_size, edges, old_to_new')
        cnt = 0
        for v in self.G.nodes:
            if np.random.binomial(1, p_seed):
                aug = Aug(seed=v, active_pnt=v, next_pnt=None, alive=True, cur_size=0,
                          max_size=np.random.choice(range(1, k + 1), 1), edges=[], old_to_new=False)
                augmentations[cnt] = aug
                self.G.nodes[v]['touched'] = True
                cnt += 1

        # First extension step
        receivers = dict()
        for idx in augmentations.keys():
            near_scheduled = False
            aug = augmentations[idx]
            for edge in self.neighbor_edges('v', aug.seed, None):
                if self.G.edges[edge]['scheduled']:
                    near_scheduled = True
                    aug.next_pnt = edge[1]
                    aug.edges.append((aug.active_pnt, aug.next_pnt))

            if not near_scheduled:
                candidates = list(self.G.adj[aug.seed])
                chosen_idx = int(np.random.choice(range(len(candidates))))

                aug.next_pnt = candidates[chosen_idx]
                aug.old_to_new = True

            if aug.next_pnt not in receivers.keys():
                receivers[aug.next_pnt] = {idx}

            else:
                receivers[aug.next_pnt].add(idx)

        # First avoiding contentions
        for r_pnt in receivers.keys():

            if self.G.nodes[r_pnt]['touched'] or len(receivers[r_pnt]) >= 2:
                for idx in receivers[r_pnt]:
                    augmentations[idx].alive = False
            else:
                idx = next(iter(receivers[r_pnt]))  # aug = next(receivers[r_pnt])
                aug = augmentations[idx]
                if aug.old_to_new:
                    aug.edges.append((aug.active_pnt, aug.next_pnt))
                    aug.cur_size = aug.cur_size + 1
                aug.old_to_new = not aug.old_to_new

        # The other steps
        for tau in range(1, 2 * k + 2):
            # For extension
            receivers = dict()
            for idx in range(cnt):
                aug = augmentations[idx]
                if aug.alive:
                    candidates = self.neighbor_edges('v', aug.next_pnt, aug.active_pnt)

                    if aug.old_to_new:
                        picked_idx = np.random.choice(range(len(candidates)))
                        picked_one = candidates[picked_idx]
                    else:
                        picked_one = None
                        for edge in candidates:
                            if self.G.edges[edge]['scheduled']:
                                picked_one = edge

                    if picked_one is None:
                        aug.alive = False
                    else:
                        aug.active_pnt = aug.next_pnt
                        aug.next_pnt = picked_one[1]
                        self.G.nodes[aug.active_pnt]['touched'] = True

                        if not aug.old_to_new:
                            aug.edges.append(picked_one)

                        if aug.next_pnt not in receivers.keys():
                            receivers[aug.next_pnt] = {idx}
                        else:
                            receivers[aug.next_pnt].add(idx)

            # For avoiding contentions
            for r_pnt in receivers.keys():

                if self.G.nodes[r_pnt]['touched'] or len(receivers[r_pnt]) >= 2:
                    for idx in receivers[r_pnt]:
                        augmentations[idx].alive = False
                else:
                    idx = next(iter(receivers[r_pnt]))  # aug = next(receivers[r_pnt])
                    aug = augmentations[idx]
                    if aug.old_to_new:
                        aug.edges.append((aug.active_pnt, aug.next_pnt))
                        aug.cur_size = aug.cur_size + 1
                    aug.old_to_new = not aug.old_to_new

        # Check the possiblity of building cycle
        for idx in augmentations.keys():
            aug = augmentations[idx]
            neighbors = set(self.G.adj[aug.active_pnt])
            if len(aug.edges) >= 2 and aug.seed in neighbors and aug.cur_size < aug.max_size:
                first_edge = aug.edges[0]
                last_edge = aug.edges[-1]
                if self.G.edges[first_edge]['scheduled'] and self.G.edges[last_edge]['scheduled']:
                    aug.edges.append((aug.active_pnt, aug.seed))
                    aug.cur_size = aug.cur_size + 1

        # Calculate gain of all augmentations
        for idx, aug in augmentations.items():
            gain = 0
            for edge in aug.edges:
                if self.G.edges[edge]['scheduled']:
                    gain -= self.G.edges[edge][estimator]
                else:
                    gain += self.G.edges[edge][estimator]

            if gain > 0:
                for edge in aug.edges:
                    self.G.edges[edge]['change'] = True

        # Switch matching according to schedule
        matching = []
        for edge in self.G.edges:
            if self.G.edges[edge]['change']:
                self.G.edges[edge]['scheduled'] = not self.G.edges[edge]['scheduled']
            if self.G.edges[edge]['scheduled']:
                matching.append(edge)

        sum_reward = 0
        for e in matching:
            sum_reward += self.G.edges[e][weight]

        return matching, sum_reward

    @staticmethod
    def matching_checker(matching):
        tested = set()
        for node1, node2 in matching:
            if node1 not in tested and node2 not in tested:
                tested.add(node1)
                tested.add(node2)
            else:
                return False
        return True
