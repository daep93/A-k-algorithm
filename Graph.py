import networkx as nx
# from tqdm import tqdm_notebook
import numpy as np
import math
from recordclass import recordclass
import matplotlib.pyplot as plt
import os

class GridGraph:
    def __init__(self, _nodes, _arrivals, _capacities):
        _capacities = iter(_capacities)
        _arrivals = iter(_arrivals)
        self.graph = nx.grid_2d_graph(_nodes, _nodes)
        self.pos = nx.spring_layout(self.graph)
        for edge in self.graph.edges:
            self.graph.edges[edge]['arrivals'] = iter(next(_arrivals))
            self.graph.edges[edge]['capacity'] = next(_capacities)
            self.graph.edges[edge]['backlog'] = 0
            self.graph.edges[edge]['backlog_snapshot'] = 0
            self.graph.edges[edge]['scheduled'] = False
            self.graph.edges[edge]['sample_mean'] = 1
            self.graph.edges[edge]['count'] = 1
            self.graph.edges[edge]['alpha'] = 1
            self.graph.edges[edge]['beta'] = 0

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

    def get_neighbors(self, obj, past):
        """
        func_name: neighbor
            This function returns the neighbor edges of 'obj'.
            the left point of all neighbor edge is obj.
        :param ty:
            Neighbor edges of edge  -> input 'edge'
            Neighbor edges of point -> input 'point'.
        :param obj:
            information of point or edge
        :param past:
            if you want not to consider 'past' point, then function will return the neighbor edges
            except edge consisting the past point.
        :return:
            list of neighbor edges of 'obj'
        """

        if isinstance(obj[0], tuple):
            end1, end2 = obj
            neighbor_1 = set([(end1, neighbor) for neighbor in self.graph.adj[end1]])
            neighbor_1.remove((end1, end2))
            neighbor_2 = set([(end2, neighbor) for neighbor in self.graph.adj[end2]])
            neighbor_2.remove((end2, end1))
            neighbors = list(neighbor_1.union(neighbor_2))

        elif isinstance(obj[0], int):
            adj = list(self.graph.adj[obj])
            if past is not None:
                adj.remove(past)
            neighbors = [(obj, neighbor) for neighbor in adj]
        else:
            raise Exception('Need proper type to \'obj\' in neighbors function')
        return neighbors

    def init_estimator(self, estimator):
        if estimator == 'capacity':
            return
        else:
            for edge in self.graph.edges:
                self.graph.edges[edge]['sample mean'] = 1
                self.graph.edges[edge]['count'] = 1
                self.graph.edges[edge]['alpha'] = 1
                self.graph.edges[edge]['beta'] = 0

    def get_estimator(self, estimator, time):
        """
        func_name: estimator_update
            At each time, this function updates estimators
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return: None
        """
        if estimator == 'TS':
            for edge in self.graph.edges:
                self.graph.edges[edge]['TS'] = self.graph.edges[edge]['queue_ratio'] * np.random.beta(
                    self.graph.edges[edge]['alpha'] + 1, self.graph.edges[edge]['beta'] + 1
                )

        elif estimator == 'UCB':
            for edge in self.graph.edges:
                self.graph.edges[edge]['UCB'] = self.graph.edges[edge]['queue_ratio'] * self.graph.edges[edge][
                    'sample mean'] \
                                                + math.sqrt(
                    (self.graph.number_of_nodes() + 1) * math.log(time) / (
                        self.graph.edges[edge]['count'])
                )
        else:
            raise Exception(f'We don\'t deal with {estimator} as estimator')

    def update_after_schedule(self, matching, estimator):
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
            raise Exception(f'We don\'t deal with {estimator} as estimator')

    def get_backlog_max(self):
        return max(1, max([self.graph.edges[edge]['backlog_snapshot'] for edge in self.graph.edges]))

    def queue_arrival(self):
        for edge in self.graph.edges():
            self.graph.edges[edge]['backlog'] += int(next(self.graph.edges[edge]['arrivals']))

    def queue_departure(self):
        for edge in self.graph.edges():
            self.graph.edges[edge]['backlog'] = max(0, self.graph.edges[edge]['backlog'] - np.random.binomial(1,self.graph.edges[edge]['capacity']))

    def queue_snapshot(self):  # Should revise this function to get 'backlog_snapshot' for one frame
        for edge in self.graph.edges:
            self.graph.edges[edge]['backlog_snapshot'] = self.graph.edges[edge]['backlog']

    # From this part, we define MWM, GMM, AUG as combinatorial algorithms for one frame
    # 1. MWM
    def max_weight_matching(self, estimator, time):
        """
        func_name: max_weight_matching
            This function calculate max weight matching of given graph and return the matching
            and mean reward of the matching.

        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return:
            return the max weight matching
        """
        # For calculation of queue ratio
        backlog_max = self.get_backlog_max()

        for edge in self.graph.edges:
            self.graph.edges[edge]['queue_ratio'] = self.graph.edges[edge]['backlog_snapshot'] / backlog_max

        # Get estimator
        self.get_estimator(estimator, time)

        # Get MWM
        matching = nx.max_weight_matching(self.graph, weight=estimator)

        return matching

    # 2. GMM
    def greedy_maximum_matching(self, estimator, time):
        """
        func_name: greedy_maximum_matching
            This function calculate greedy maximum weight matching of given graph
            and return the matching and mean reward of the matching.

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

        # For calculation of queue ratio
        backlog_max = self.get_backlog_max()

        for edge in self.graph.edges:
            self.graph.edges[edge]['queue_ratio'] = self.graph.edges[edge]['backlog_snapshot'] / backlog_max

        # Get estimator
        self.get_estimator(estimator, time)

        # Get GMM
        # 1. Sort edges in order of largest estimator
        edges = [Edge(edge=e, weight=self.graph.edges[e][estimator]) for e in self.graph.edges]
        sorted_edges = [e.edge for e in sorted(edges, key=lambda l: l.weight, reverse=True)]

        # 2. Greedy Process with sorted edges
        matching = []
        while len(sorted_edges) != 0:
            next_edge = sorted_edges.pop(0)
            matching.append(next_edge)

            adj_edges = self.get_neighbors(next_edge, None)

            for target in adj_edges:
                reversed_target = tuple(reversed(target))

                if target in sorted_edges:
                    sorted_edges.remove(target)
                elif reversed_target in sorted_edges:
                    sorted_edges.remove(reversed_target)

        return matching

    # 3. AUG
    def refresh_graph(self):
        """
        During 4k+2, A^k algorithm leaves traces of extending
        Initialize 'change' variable and touched trace for next slot
        :return: None
        """
        for edge in self.graph.edges:
            self.graph.edges[edge]['change'] = False

        for _node in self.graph.nodes:
            self.graph.nodes[_node]['touched'] = False

    def A_k_matching(self, k, p_seed, estimator, time):
        """
        func_name: augmentation
                    This function calculate a matching by augmenting
                    and return the matching and mean reward of the matching.
        :param k:
            intended size (= upper bound of augmentation size)
        :param p_seed:
            the probability of seed
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return:
            return the calculated matching by augmenting and mean reward of the matching

        """
        # For calculation of queue ratio
        backlog_max = self.get_backlog_max()

        for edge in self.graph.edges:
            self.graph.edges[edge]['queue_ratio'] = self.graph.edges[edge]['backlog_snapshot'] / backlog_max

        # Get estimator
        self.get_estimator(estimator, time)

        # Refresh graph at every time slot
        self.refresh_graph()

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
            for edge in self.get_neighbors(aug.seed, None):
                if self.graph.edges[edge]['scheduled']:
                    near_scheduled = True
                    aug.next_pnt = edge[1]
                    aug.edges.append((aug.active_pnt, aug.next_pnt))
                    self.graph.nodes[aug.next_pnt]['touched'] = True
                    break

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
                idx = (receivers[r_pnt]).pop()  # aug = next(receivers[r_pnt])
                aug = augmentations[idx]
                if aug.old_to_new:
                    aug.edges.append((aug.active_pnt, aug.next_pnt))
                    aug.cur_size = aug.cur_size + 1
                aug.old_to_new = not aug.old_to_new
                self.graph.nodes[r_pnt]['touched'] = True

        # The other steps
        for tau in range(2, 2 * k + 2):
            # For extension
            receivers = dict()
            for idx in range(cnt):
                aug = augmentations[idx]
                if aug.old_to_new and aug.cur_size == aug.max_size:
                    aug.alive = False
                if aug.alive:
                    candidates = self.get_neighbors(aug.next_pnt, aug.active_pnt)

                    if aug.old_to_new:
                        picked_idx = np.random.choice(range(len(candidates)))
                        picked_one = candidates[picked_idx]
                        while self.graph.edges[picked_one]['scheduled']:
                            candidates.remove(picked_one)
                            if len(candidates):
                                picked_idx = np.random.choice(range(len(candidates)))
                                picked_one = np.random.choice(candidates)
                            else:
                                picked_one = None
                                break
                    else:
                        picked_one = None
                        for edge in candidates:
                            if self.graph.edges[edge]['scheduled']:
                                picked_one = edge
                                break

                    if picked_one is None:
                        aug.alive = False
                    else:
                        aug.active_pnt = aug.next_pnt
                        aug.next_pnt = picked_one[1]
                        if not aug.old_to_new:
                            self.graph.nodes[aug.active_pnt]['touched'] = True
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
                    idx = (receivers[r_pnt]).pop()  # aug = next(receivers[r_pnt])
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
            print('No matching!')
            print(augmentations)
            self.draw(past_matching, matching)
            raise Exception('Please revise UCB algorithm')

        return matching
class LargeGraph:
    def __init__(self, _nodes, _edges, _arrivals, _capacities):
        _capacities = iter(_capacities)
        _arrivals = iter(_arrivals)
        self.graph = nx.dense_gnm_random_graph(_nodes, _edges)
        self.pos = nx.spring_layout(self.graph)
        for edge in self.graph.edges:
            self.graph.edges[edge]['arrivals'] = iter(next(_arrivals))
            self.graph.edges[edge]['capacity'] = next(_capacities)
            self.graph.edges[edge]['backlog'] = 0
            self.graph.edges[edge]['backlog_snapshot'] = 0
            self.graph.edges[edge]['scheduled'] = False
            self.graph.edges[edge]['sample_mean'] = 1
            self.graph.edges[edge]['count'] = 1
            self.graph.edges[edge]['alpha'] = 1
            self.graph.edges[edge]['beta'] = 0
            
    def reinit(self):
        for edge in self.graph.edges:
            self.graph.edges[edge]['backlog'] = 0
            self.graph.edges[edge]['backlog_snapshot'] = 0
            self.graph.edges[edge]['scheduled'] = False
        self.refresh_graph() 
        
    def reset_environment(self, edges, min_lambda, max_lambda, min_mu, max_mu, frame_size, frame_rep):
        path1 = './arrival_patterns'
        arrival_list = os.listdir(path1)
        
        with open(
                './arrival_patterns/' + f'Arrivals(edge {edges}, lambda from {min_lambda:.4f} to {max_lambda:.4f}, frame_size {frame_size}, frame_rep {frame_rep}).txt',
            'r') as f:
            arrivals = f.readlines()
        arrivals = iter(arrivals)
        
        for edge in self.graph.edges:
            self.graph.edges[edge]['arrivals'] = iter(next(arrivals))
    
    def draw(self, *matchings):
        num = len(matchings)
        cnt = 1
        for matching in matchings:
            plt.subplot('1' + str(num) + str(cnt))
            cnt += 1

            nx.draw(self.graph, pos=nx.spring_layout(self.graph), with_labels=True)
            # nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.graph.nodes, node_color='r', node_size=1000, alpha=0.5)
            # nx.draw_networkx_edge_labels(self.graph, self.pos,
            #                              edge_labels=nx.get_edge_attributes(self.graph, 'capacity'))
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=matching, width=8, alpha=0.5, edge_color='b')
        plt.show()

    def get_neighbors(self, obj, past=None):
        """
        func_name: neighbor
            This function returns the neighbor edges of 'obj'.
            the left point of all neighbor edge is obj.
        :param ty:
            Neighbor edges of edge  -> input 'edge'
            Neighbor edges of point -> input 'point'.
        :param obj:
            information of point or edge
        :param past:
            if you want not to consider 'past' point, then function will return the neighbor edges
            except edge consisting the past point.
        :return:
            list of neighbor edges of 'obj'
        """

        if isinstance(obj, tuple):
            end1, end2 = obj
            neighbor_1 = set([(end1, neighbor) for neighbor in self.graph.adj[end1]])
            neighbor_1.remove((end1, end2))
            neighbor_2 = set([(end2, neighbor) for neighbor in self.graph.adj[end2]])
            neighbor_2.remove((end2, end1))
            neighbors = list(neighbor_1.union(neighbor_2))

        elif isinstance(obj, int):
            adj = list(self.graph.adj[obj])
            if past is not None:
                adj.remove(past)
            neighbors = [(obj, neighbor) for neighbor in adj]
        else:
            raise Exception('Need proper type to \'obj\' in neighbors function')
        
        return neighbors

    def init_estimator(self, estimator):
        if estimator == 'capacity':
            return
        else:
            for edge in self.graph.edges:
                self.graph.edges[edge]['sample mean'] = 1
                self.graph.edges[edge]['count'] = 1
                self.graph.edges[edge]['alpha'] = 1
                self.graph.edges[edge]['beta'] = 0

    def get_estimator(self, estimator, time):
        """
        func_name: estimator_update
            At each time, this function updates estimators
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return: None
        """
        if estimator == 'TS':
            for edge in self.graph.edges:
                self.graph.edges[edge]['TS'] = self.graph.edges[edge]['queue_ratio'] * np.random.beta(
                    self.graph.edges[edge]['alpha'] + 1, self.graph.edges[edge]['beta'] + 1
                )

        elif estimator == 'UCB':
            for edge in self.graph.edges:
                self.graph.edges[edge]['UCB'] = self.graph.edges[edge]['queue_ratio'] * self.graph.edges[edge][
                    'sample mean'] \
                                                + math.sqrt(
                    (self.graph.number_of_nodes() + 1) * math.log(time) / (
                        self.graph.edges[edge]['count'])
                )
        else:
            raise Exception(f'We don\'t deal with {estimator} as estimator')

    def get_estimator(self, edge, local_max_backlog,  estimator, time):
        """
        func_name: estimator_update
            At each time, this function updates estimators
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return: None
        """
        if local_max_backlog:
            queue_ratio = self.graph.edges[edge]['backlog_snapshot']/local_max_backlog
        else:
            queue_ratio = 1
            
        if estimator == 'TS':
            self.graph.edges[edge]['TS'] =  queue_ratio * np.random.beta(
                self.graph.edges[edge]['alpha'] + 1, self.graph.edges[edge]['beta'] + 1
            )

        elif estimator == 'UCB':
            
            self.graph.edges[edge]['UCB'] = queue_ratio * self.graph.edges[edge][
                'sample mean'] + math.sqrt((self.graph.number_of_nodes() + 1) * math.log(time) / (self.graph.edges[edge]['count']))
        else:
            raise Exception(f'We don\'t deal with {estimator} as estimator')        
            
            
    def update_after_schedule(self, matching, estimator):
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
            raise Exception(f'We don\'t deal with {estimator} as estimator')

    def get_backlog_max(self):
        return max(1, max([self.graph.edges[edge]['backlog_snapshot'] for edge in self.graph.edges]))

    def queue_arrival(self):
        for edge in self.graph.edges():
            self.graph.edges[edge]['backlog'] += int(next(self.graph.edges[edge]['arrivals']))

    def queue_departure(self):
        for edge in self.graph.edges():
            self.graph.edges[edge]['backlog'] = max(0, self.graph.edges[edge]['backlog'] - np.random.binomial(1,self.graph.edges[edge]['capacity']))

    def queue_snapshot(self):  # Should revise this function to get 'backlog_snapshot' for one frame
        for edge in self.graph.edges:
            self.graph.edges[edge]['backlog_snapshot'] = self.graph.edges[edge]['backlog']

    # From this part, we define MWM, GMM, AUG as combinatorial algorithms for one frame
    # 1. MWM
    def max_weight_matching(self, estimator, time):
        """
        func_name: max_weight_matching
            This function calculate max weight matching of given graph and return the matching
            and mean reward of the matching.

        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return:
            return the max weight matching
        """
        # For calculation of queue ratio
        backlog_max = self.get_backlog_max()

        for edge in self.graph.edges:
            self.graph.edges[edge]['queue_ratio'] = self.graph.edges[edge]['backlog_snapshot'] / backlog_max

        # Get estimator
        self.get_estimator(estimator, time)

        # Get MWM
        matching = nx.max_weight_matching(self.graph, weight=estimator)

        return matching

    # 2. GMM
    def greedy_maximum_matching(self, estimator, time):
        """
        func_name: greedy_maximum_matching
            This function calculate greedy maximum weight matching of given graph
            and return the matching and mean reward of the matching.

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

        # For calculation of queue ratio
        backlog_max = self.get_backlog_max()

        for edge in self.graph.edges:
            self.graph.edges[edge]['queue_ratio'] = self.graph.edges[edge]['backlog_snapshot'] / backlog_max

        # Get estimator
        self.get_estimator(estimator, time)

        # Get GMM
        # 1. Sort edges in order of largest estimator
        edges = [Edge(edge=e, weight=self.graph.edges[e][estimator]) for e in self.graph.edges]
        sorted_edges = [e.edge for e in sorted(edges, key=lambda l: l.weight, reverse=True)]

        # 2. Greedy Process with sorted edges
        matching = []
        while len(sorted_edges) != 0:
            next_edge = sorted_edges.pop(0)
            matching.append(next_edge)

            adj_edges = self.get_neighbors(next_edge, None)

            for target in adj_edges:
                reversed_target = tuple(reversed(target))

                if target in sorted_edges:
                    sorted_edges.remove(target)
                elif reversed_target in sorted_edges:
                    sorted_edges.remove(reversed_target)

        return matching

    # 3. AUG
    def refresh_graph(self):
        """
        During 4k+2, A^k algorithm leaves traces of extending
        Initialize 'change' variable and touched trace for next slot
        :return: None
        """
        for edge in self.graph.edges:
            self.graph.edges[edge]['change'] = False

        for _node in self.graph.nodes:
            self.graph.nodes[_node]['touched'] = False

    def A_k_matching(self, k, p_seed, estimator, time):
        """
        func_name: augmentation
                    This function calculate a matching by augmenting
                    and return the matching and mean reward of the matching.
        :param k:
            intended size (= upper bound of augmentation size)
        :param p_seed:
            the probability of seed
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return:
            return the calculated matching by augmenting and mean reward of the matching

        """
        # For calculation of queue ratio
        backlog_max = self.get_backlog_max()

        for edge in self.graph.edges:
            self.graph.edges[edge]['queue_ratio'] = self.graph.edges[edge]['backlog_snapshot'] / backlog_max

        # Get estimator
        self.get_estimator(estimator, time)

        # Refresh graph at every time slot
        self.refresh_graph()

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
            for edge in self.get_neighbors(aug.seed, None):
                if self.graph.edges[edge]['scheduled']:
                    near_scheduled = True
                    aug.next_pnt = edge[1]
                    aug.edges.append((aug.active_pnt, aug.next_pnt))
                    self.graph.nodes[aug.next_pnt]['touched'] = True
                    break

            if not near_scheduled:
                candidates = list(self.graph.adj[aug.seed])
                aug.next_pnt = int(np.random.choice(candidates))

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
                idx = receivers[r_pnt].pop()  # aug = next(receivers[r_pnt])
                aug = augmentations[idx]
                if aug.old_to_new:
                    aug.edges.append((aug.active_pnt, aug.next_pnt))
                    aug.cur_size = aug.cur_size + 1
                aug.old_to_new = not aug.old_to_new
                self.graph.nodes[r_pnt]['touched'] = True

        # The other steps
        for tau in range(2, 2 * k + 2):
            # For extension
            receivers = dict()
            for idx in range(cnt):
                aug = augmentations[idx]
                if aug.old_to_new and aug.cur_size == aug.max_size:
                    aug.alive = False
                if aug.alive:
                    candidates = self.get_neighbors(aug.next_pnt, aug.active_pnt)

                    if aug.old_to_new:
                        picked_one = np.random.choice(candidates)
                        while self.graph.edges[picked_one]['scheduled']:
                            candidates.remove(picked_one)
                            if len(candidates):
                                picked_one = np.random.choice(candidates)
                            else:
                                picked_one = None
                                break
                                
                    else:
                        picked_one = None
                        for edge in candidates:
                            if self.graph.edges[edge]['scheduled']:
                                picked_one = edge
                                break

                    if picked_one is None:
                        aug.alive = False
                    else:
                        aug.active_pnt = aug.next_pnt
                        aug.next_pnt = picked_one[1]
                        if not aug.old_to_new:
                            self.graph.nodes[aug.next_pnt]['touched'] = True
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
                    idx = (receivers[r_pnt]).pop()  
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
            print('No matching!')
            print(augmentations)
            self.draw(past_matching, matching)
            raise Exception('Please revise UCB algorithm')

        return matching

    def dA_k_matching(self, k, p_seed, estimator, time):
        """
        func_name: augmentation
                    This function calculate a matching by real distributed augmenting
                    and return the matching and mean reward of the matching.
        :param k:
            intended size (= upper bound of augmentation size)
        :param p_seed:
            the probability of seed
        :param estimator:
            Thompson sampling -> 'TS'
            Upper Confidence Bound -> 'UCB'
            not want to use estimator -> the other command
        :param time:
            time slot
        :return:
            return the calculated matching by augmenting and mean reward of the matching

        """
        # For calculation of local queue ratio
        for node in self.graph.nodes:
            self.graph.nodes[node]['local_max_backlog'] = max([self.graph.edges[neighbor]['backlog_snapshot'] for neighbor in self.get_neighbors(node)]) 

        

        # Refresh graph at every time slot
        self.refresh_graph()

        # Generate seeds for augmenting
        augmentations = dict()
        Aug = recordclass('Augmentation', 'seed, active_pnt, next_pnt, alive, cur_size, max_size, edges,local_max_backlog, old_to_new')
        cnt = 0
        for v in self.graph.nodes:
            if np.random.binomial(1, p_seed):
                aug = Aug(seed=v, active_pnt=v, next_pnt=None, alive=True, cur_size=0,
                          max_size=int(np.random.choice(range(1, k + 1), 1)), edges=[], local_max_backlog=0, old_to_new=False)
                augmentations[cnt] = aug
                self.graph.nodes[v]['touched'] = True
                cnt += 1

        # First extension step
        receivers = dict()
        for idx in augmentations.keys():
            near_scheduled = False
            aug = augmentations[idx]
            for edge in self.get_neighbors(aug.seed, None):
                if self.graph.edges[edge]['scheduled']:
                    near_scheduled = True
                    aug.next_pnt = edge[1]
                    aug.edges.append((aug.active_pnt, aug.next_pnt))
                    self.graph.nodes[aug.next_pnt]['touched'] = True
                    break

            if not near_scheduled:
                candidates = list(self.graph.adj[aug.seed])
                aug.next_pnt = int(np.random.choice(candidates))

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
                idx = receivers[r_pnt].pop()  # aug = next(receivers[r_pnt])
                aug = augmentations[idx]
                if aug.old_to_new:
                    aug.edges.append((aug.active_pnt, aug.next_pnt))
                    aug.cur_size = aug.cur_size + 1
                aug.old_to_new = not aug.old_to_new
                self.graph.nodes[r_pnt]['touched'] = True

        # The other steps
        for tau in range(2, 2 * k + 2):
            # For extension
            receivers = dict()
            for idx in range(cnt):
                aug = augmentations[idx]
                if aug.old_to_new and aug.cur_size == aug.max_size:
                    aug.alive = False
                if aug.alive:
                    candidates = self.get_neighbors(aug.next_pnt, aug.active_pnt)

                    if aug.old_to_new:
                        picked_one = np.random.choice(candidates)
                        while self.graph.edges[picked_one]['scheduled']:
                            candidates.remove(picked_one)
                            if len(candidates):
                                picked_one = np.random.choice(candidates)
                            else:
                                picked_one = None
                                break
                                
                    else:
                        picked_one = None
                        for edge in candidates:
                            if self.graph.edges[edge]['scheduled']:
                                picked_one = edge
                                break

                    if picked_one is None:
                        aug.alive = False
                    else:
                        aug.active_pnt = aug.next_pnt
                        aug.next_pnt = picked_one[1]
                        if not aug.old_to_new:
                            self.graph.nodes[aug.next_pnt]['touched'] = True
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
                    idx = (receivers[r_pnt]).pop()  
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
        
        # Get queue ratio
        for idx in augmentations.keys():
            aug = augmentations[idx]
            tmp_max = -1
            for act_pnt, next_pnt in aug.edges:
                edge_max = max(self.graph.nodes[act_pnt]['local_max_backlog'],self.graph.nodes[next_pnt]['local_max_backlog'])
                if tmp_max < edge_max:
                    tmp_max = edge_max
            aug.local_max_backlog = tmp_max
                

        # Calculate gain of all augmentations
        for idx, aug in augmentations.items():
            gain = 0
            for edge in aug.edges:
                self.get_estimator(edge,aug.local_max_backlog, estimator, time)
                if self.graph.edges[edge]['scheduled']:
                    gain -= self.graph.edges[edge][estimator]
                else:
                    gain += self.graph.edges[edge][estimator]

            if gain > 0:
                for edge in aug.edges:
                    self.graph.edges[edge]['change'] = True
        
        # Update local_max_queue
        for idx in augmentations.keys():
            aug = augmentations[idx]
            for act_pnt, next_pnt in aug.edges:
                self.graph.nodes[act_pnt]['local_max_backlog'] = aug.local_max_backlog
                self.graph.nodes[next_pnt]['local_max_backlog'] = aug.local_max_backlog
        
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
            print('No matching!')
            print(augmentations)
            self.draw(past_matching, matching)
            raise Exception('Please revise UCB algorithm')

        return matching    
    

def matching_checker(matching):
    tested = set()
    for node1, node2 in matching:
        if node1 not in tested and node2 not in tested:
            tested.add(node1)
            tested.add(node2)
        else:
            return False
    return True
