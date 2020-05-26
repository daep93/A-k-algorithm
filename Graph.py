import networkx as nx
# from tqdm import tqdm_notebook
import numpy as np
import math
from recordclass import recordclass
import matplotlib.pyplot as plt


class Gridgraph:

    def __init__(self, _nodes, _capacities):
        _capacities = iter(_capacities)
        self.graph = nx.grid_2d_graph(_nodes, _nodes)
        self.pos = nx.spring_layout(self.graph)
        for edge in self.graph.edges:
            self.graph.edges[edge]['capacity'] = next(_capacities)
            self.graph.edges[edge]['backlog'] = 0
            self.graph.edges[edge]['tmp_backlog'] = 0
            self.graph.edges[edge]['scheduled'] = False
            self.graph.edges[edge]['sample mean'] = 1
            self.graph.edges[edge]['count'] = 1
            self.graph.edges[edge]['alpha'] = 1
            self.graph.edges[edge]['beta'] = 0

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
            neighbor_1 = set([(end1, neighbor) for neighbor in self.graph.adj[end1]])
            neighbor_1.remove((end1, end2))
            neighbor_2 = set([(end2, neighbor) for neighbor in self.graph.adj[end2]])
            neighbor_2.remove((end2, end1))

            neighbors = list(neighbor_1.union(neighbor_2))
        else:
            adj = list(self.graph.adj[obj])
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

            nx.draw(self.graph, self.pos, with_labels=True)
            # nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.graph.nodes, node_color='r', node_size=1000, alpha=0.5)
            # nx.draw_networkx_edge_labels(self.graph, self.pos,
            #                              edge_labels=nx.get_edge_attributes(self.graph, 'capacity'))
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=matching, width=8, alpha=0.5, edge_color='b')
        plt.show()

    def refresh(self):
        """
        Only for augmentation algorithm.
        Initialize 'change' variable and touched trace
        :return: None
        """
        for edge in self.graph.edges:
            self.graph.edges[edge]['change'] = False

        for _node in self.graph.nodes:
            self.graph.nodes[_node]['touched'] = False

    def queue_arrive(self, edge, arrive):
        self.graph.edges[edge]['tmp_backlog'] += arrive

    def queue_departure(self, edge):
        self.graph.edges[edge]['tmp_backlog'] = max(0, self.graph.edges[edge]['tmp_backlog']-1)

    def queue_update(self):
        for edge in self.graph.edges:
            self.graph.edges[edge]['backlog'] = self.graph.edges[edge]['tmp_backlog']

    def after_schedule_update(self, matching, estimator):
        if estimator == 'UCB':
            for edge in matching:
                x = np.random.binomial(1, self.graph.edges[edge]['capacity'])
                self.graph.edges[edge]['sample mean'] = self.graph.edges[edge]['sample mean'] * self.graph.edges[edge][
                    'count'] + x
                self.graph.edges[edge]['count'] = self.graph.edges[edge]['count'] + 1
                self.graph.edges[edge]['sample mean'] = self.graph.edges[edge]['sample mean'] / self.graph.edges[edge][
                    'count']
        elif estimator == 'TS':
            for edge in matching:
                x = np.random.binomial(1, self.graph.edges[edge]['capacity'])
                self.graph.edges[edge]['alpha'] = self.graph.edges[edge]['alpha'] + x
                self.graph.edges[edge]['beta'] = self.graph.edges[edge]['beta'] + 1 - x
        else:
            return

    def estimaotr_init(self, estimator):
        if estimator == 'capacity':
            return
        else:
            for edge in self.graph.edges:
                self.graph.edges[edge]['sample mean'] = 1
                self.graph.edges[edge]['count'] = 1
                self.graph.edges[edge]['alpha'] = 1
                self.graph.edges[edge]['beta'] = 0

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
                for edge in self.graph.edges:
                    self.graph.edges[edge]['TS'] = self.graph.edges[edge]['weight'] * np.random.beta(
                        self.graph.edges[edge]['alpha'] + 1, self.graph.edges[edge]['beta'] + 1
                    )
            else:
                for edge in self.graph.edges:
                    self.graph.edges[edge]['TS'] = np.random.beta(
                        self.graph.edges[edge]['alpha'] + 1, self.graph.edges[edge]['beta'] + 1
                    )

        elif estimator == 'UCB':
            if with_queue:
                for edge in self.graph.edges:
                    self.graph.edges[edge]['UCB'] = self.graph.edges[edge]['weight'] * self.graph.edges[edge][
                        'sample mean'] \
                                                    + math.sqrt(
                        (self.graph.number_of_nodes() + 1) * math.log(time) / (
                            self.graph.edges[edge]['count'])
                    )
            else:
                for edge in self.graph.edges:
                    self.graph.edges[edge]['UCB'] = self.graph.edges[edge]['sample mean'] \
                                                    + math.sqrt(
                        (self.graph.number_of_nodes() + 1) * math.log(time) / (
                            self.graph.edges[edge]['count'])
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
            backlogs_snapshot = set([self.graph.edges[edge]['backlog'] for edge in self.graph.edges])
            backlog_max = max(backlogs_snapshot)

            weight = 'weighted capacity'
            if estimator == 'capacity':
                estimator = weight

            if backlog_max == 0:
                backlog_max = 1

            for edge in self.graph.edges:
                self.graph.edges[edge]['weight'] = self.graph.edges[edge]['backlog'] / backlog_max
                self.graph.edges[edge][weight] = self.graph.edges[edge]['weight'] * self.graph.edges[edge]['capacity']

        self.estimator_update(estimator, with_queue, time)
        matching = nx.max_weight_matching(self.graph, weight=estimator)

        sum_reward = 0
        for edge in matching:
            sum_reward += self.graph.edges[edge][weight]

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
            backlogs_snapshot = set([self.graph.edges[edge]['backlog'] for edge in self.graph.edges])
            backlog_max = max(backlogs_snapshot)

            weight = 'weighted capacity'
            if estimator == 'capacity':
                estimator = weight

            if backlog_max == 0:
                backlog_max = 1

            for edge in self.graph.edges:
                self.graph.edges[edge]['weight'] = self.graph.edges[edge]['backlog'] / backlog_max
                self.graph.edges[edge][weight] = self.graph.edges[edge]['weight'] * self.graph.edges[edge]['capacity']

        edges = [Edge(edge=e, weight=self.graph.edges[e][estimator]) for e in self.graph.edges]
        sorted_edges = [e.edge for e in sorted(edges, key=lambda l: l.weight, reverse=True)]

        matching = []
        while len(sorted_edges) != 0:

            next_edge = sorted_edges[0]

            matching.append(next_edge)
            sorted_edges.remove(next_edge)

            edge_adj_list = self.neighbor_edges('e', next_edge, None)

            for target in edge_adj_list:
                reversed_target = tuple(reversed(target))

                if target in sorted_edges:
                    print('delete', target)
                    sorted_edges.remove(target)
                if reversed_target in sorted_edges:
                    print('delete', reversed_target)
                    sorted_edges.remove(reversed_target)
        sum_reward = 0
        for edge in matching:
            sum_reward += self.graph.edges[edge][weight]

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
            backlogs_snapshot = set([self.graph.edges[edge]['backlog'] for edge in self.graph.edges])
            backlog_max = max(backlogs_snapshot)

            weight = 'weighted capacity'
            if estimator == 'capacity':
                estimator = weight

            if backlog_max == 0:
                backlog_max = 1

            for edge in self.graph.edges:
                self.graph.edges[edge]['weight'] = self.graph.edges[edge]['backlog'] / backlog_max
                self.graph.edges[edge][weight] = self.graph.edges[edge]['weight'] * self.graph.edges[edge]['capacity']

        # Update estimator and refresh scheduled
        self.estimator_update(estimator, with_queue, time)
        self.refresh()

        # Generate seeds for augmenting
        augmentations = dict()
        Aug = recordclass('Augmentation', 'seed, active_pnt, next_pnt, alive, cur_size, max_size, edges, old_to_new')
        cnt = 0
        for v in self.graph.nodes:
            if np.random.binomial(1, p_seed):
                aug = Aug(seed=v, active_pnt=v, next_pnt=None, alive=True, cur_size=0,
                          max_size=int(np.random.choice(range(1, k + 1), 1)), edges=[], old_to_new=False)
                augmentations[cnt] = aug
                self.graph.nodes[v]['touched'] = True
                cnt += 1

        # First extension step
        receivers = dict()
        for idx in augmentations.keys():
            near_scheduled = False
            aug = augmentations[idx]
            for edge in self.neighbor_edges('v', aug.seed, None):
                if self.graph.edges[edge]['scheduled']:
                    near_scheduled = True
                    aug.next_pnt = edge[1]
                    aug.edges.append((aug.active_pnt, aug.next_pnt))

            if not near_scheduled:
                candidates = list(self.graph.adj[aug.seed])
                chosen_idx = int(np.random.choice(range(len(candidates))))

                aug.next_pnt = candidates[chosen_idx]
                aug.old_to_new = True

            if aug.next_pnt not in receivers.keys():
                receivers[aug.next_pnt] = {idx}

            else:
                receivers[aug.next_pnt].add(idx)

        # First avoiding contentions
        for r_pnt in receivers.keys():

            if self.graph.nodes[r_pnt]['touched'] or len(receivers[r_pnt]) >= 2:
                for idx in receivers[r_pnt]:
                    augmentations[idx].alive = False
            else:
                idx = next(iter(receivers[r_pnt]))  # aug = next(receivers[r_pnt])
                aug = augmentations[idx]
                if aug.old_to_new:
                    aug.edges.append((aug.active_pnt, aug.next_pnt))
                    aug.cur_size = aug.cur_size + 1
                aug.old_to_new = not aug.old_to_new
                self.graph.nodes[r_pnt]['touched'] = True

        # The other steps
        for tau in range(1, 2 * k + 2):
            # For extension
            receivers = dict()
            for idx in range(cnt):
                aug = augmentations[idx]
                if aug.old_to_new and aug.cur_size == aug.max_size:
                    aug.alive = False
                if aug.alive:
                    candidates = self.neighbor_edges('v', aug.next_pnt, aug.active_pnt)

                    if aug.old_to_new:
                        picked_idx = np.random.choice(range(len(candidates)))
                        picked_one = candidates[picked_idx]
                    else:
                        picked_one = None
                        for edge in candidates:
                            if self.graph.edges[edge]['scheduled']:
                                picked_one = edge

                    if picked_one is None:
                        aug.alive = False
                    else:
                        aug.active_pnt = aug.next_pnt
                        aug.next_pnt = picked_one[1]
                        self.graph.nodes[aug.active_pnt]['touched'] = True

                        if not aug.old_to_new:
                            aug.edges.append(picked_one)

                        if aug.next_pnt not in receivers.keys():
                            receivers[aug.next_pnt] = {idx}
                        else:
                            receivers[aug.next_pnt].add(idx)

            # For avoiding contentions
            for r_pnt in receivers.keys():

                if self.graph.nodes[r_pnt]['touched'] or len(receivers[r_pnt]) >= 2:
                    for idx in receivers[r_pnt]:
                        augmentations[idx].alive = False
                else:
                    idx = next(iter(receivers[r_pnt]))  # aug = next(receivers[r_pnt])
                    aug = augmentations[idx]
                    if aug.old_to_new:
                        aug.edges.append((aug.active_pnt, aug.next_pnt))
                        aug.cur_size = aug.cur_size + 1
                    aug.old_to_new = not aug.old_to_new
                    self.graph.nodes[r_pnt]['touched'] = True

        # Check the possiblity of building cycle
        for idx in augmentations.keys():
            aug = augmentations[idx]
            neighbors = set(self.graph.adj[aug.active_pnt])
            if len(aug.edges) >= 2 and aug.seed in neighbors and aug.cur_size < aug.max_size:
                first_edge = aug.edges[0]
                last_edge = aug.edges[-1]
                if self.graph.edges[first_edge]['scheduled'] and self.graph.edges[last_edge]['scheduled']:
                    aug.edges.append((aug.active_pnt, aug.seed))
                    aug.cur_size = aug.cur_size + 1

        # Calculate gain of all augmentations
        for idx, aug in augmentations.items():
            gain = 0
            for edge in aug.edges:
                if self.graph.edges[edge]['scheduled']:
                    gain -= self.graph.edges[edge][estimator]
                else:
                    gain += self.graph.edges[edge][estimator]

            if gain > 0:
                for edge in aug.edges:
                    self.graph.edges[edge]['change'] = True

        # Switch matching according to schedule
        past_matching = []
        matching = []
        for edge in self.graph.edges:
            if self.graph.edges[edge]['scheduled']:
                past_matching.append(edge)

            if self.graph.edges[edge]['change']:
                self.graph.edges[edge]['scheduled'] = not self.graph.edges[edge]['scheduled']
            if self.graph.edges[edge]['scheduled']:
                matching.append(edge)

        if not matching_checker(matching):
            print()
            print('No matching!')
            print(augmentations)
            self.draw(past_matching,matching)


        sum_reward = 0
        for e in matching:
            sum_reward += self.graph.edges[e][weight]

        return matching, sum_reward


def matching_checker(matching):
    tested = set()
    for node1, node2 in matching:
        if node1 not in tested and node2 not in tested:
            tested.add(node1)
            tested.add(node2)
        else:
            return False
    return True
