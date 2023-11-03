"""
graphenv.py

Created on Fri Oct 20 2023

@author: Lukas

This file contains all code for setting up the optimal transport problem.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


class GraphEnv:
    def __init__(self, graph_size: int):
        self.graph_size = graph_size
        self.graph = nx.grid_2d_graph(graph_size, graph_size)
        self.upgraded_edges = []
        self.transport_plans = []
        self.transport_costs = []

        # set the node weights of the graph to be 1
        for node in self.graph.nodes:
            self.graph.nodes[node]['weight'] = 1

        # set the edge weights of the graph to be 1
        for edge in self.graph.edges:
            self.graph.edges[edge]['weight'] = 1


    def create_transport_plans(self, mean_1: np.ndarray, mean_2: np.ndarray,
                                 var_1: np.ndarray, var_2: np.ndarray, num_plans: int) -> None:
        """
        Creates transport plans for the optimal transport problem.

        :param mean_1: Mean of the first Gaussian distribution.
        :param mean_2: Mean of the second Gaussian distribution.
        :param var_1: Variance of the first Gaussian distribution.
        :param var_2: Variance of the second Gaussian distribution.
        :param num_plans: Number of transport plans to create.
        :return: A list of transport plans.
        """
        # assert that mean_1, mean_2, var_1, var_2 are 2D vectors
        assert len(mean_1) == 2
        assert len(mean_2) == 2
        assert len(var_1) == 2
        assert len(var_2) == 2

        plans = []
        i = 0

        # sample a point by taking the mean plus a random number from a Gaussian distribution with the given variance
        while i < num_plans: 
            source_mean = mean_1
            source_perturbation = np.random.normal(0, var_1)
            source_node = source_mean + [np.int(elem) for elem in source_perturbation]

            target_mean = mean_2
            target_perturbation = np.random.normal(0, var_2)
            target_node = target_mean + [np.int(elem) for elem in target_perturbation]

            # convert source and target node to tuples
            source_node = tuple(source_node)
            target_node = tuple(target_node)

            # if source and target node are in the graph, append the transport plan to the list of transport plans
            if source_node in self.graph.nodes and target_node in self.graph.nodes:
                plans.append((source_node, target_node))
                i += 1
        
        self.transport_plans = plans

        # set the node weights of the graph to be the number of transport plans that start or end at the node
        for node in self.graph.nodes:
            self.graph.nodes[node]['weight'] = 0

        for plan in self.transport_plans:
            self.graph.nodes[plan[0]]['weight'] += 1
            self.graph.nodes[plan[1]]['weight'] += 1
    

    def compute_transport_cost(self, transport_plans: list, preference: str='min_dist') -> float:
        """
        Computes the transport cost of the given transport plans.

        :param transport_plans: A list of transport plans.
        :param preference: The preference of the agent.
        :return: The transport cost of the given transport plans.
        """
        assert preference in ['min_dist', 'frechet_min_dist'], "Preference not supported. Preference must be 'min_dist'."

        # compute the transport cost of the given transport plans
        transport_cost = 0
        for plan in transport_plans:
            if preference == 'min_dist':
                transport_cost += self.__compute_min_dist(plan)
            elif preference == 'frechet_min_dist':
                transport_cost += self.__compute_frechet_min_dist(plan)
            else:
                raise NotImplementedError
            
        return transport_cost
    

    def __compute_min_dist(self, transport_plan: tuple) -> float:
        """
        Computes the minimum distance between the source and target node of the given transport plan.

        :param transport_plan: A transport plan.
        :return: The minimum distance between the source and target node of the given transport plan.
        """
        # assert that transport_plan is a tuple
        assert isinstance(transport_plan, tuple)

        # compute the minimum distance between the source and target node of the given transport plan
        source_node = transport_plan[0]
        target_node = transport_plan[1]

        # compute the shortest path between the source and target node
        shortest_path = nx.shortest_path(self.graph, source_node, target_node)
        min_dist = sum([self.graph.edges[edge]['weight'] for edge in zip(shortest_path[:-1], shortest_path[1:])])

        return min_dist
    

    def __compute_frechet_min_dist(self, transport_plan: tuple) -> float:
        """
        Computes the minimum distance between the source and target node of the given transport plan
        and multiplies the result with a random number sampled from a Frehet distribution with alpha=8.

        :param transport_plan: A transport plan.
        :return: The minimum distance between the source and target node of the given transport plan.
        """
        # assert that transport_plan is a tuple
        assert isinstance(transport_plan, tuple)

        # compute the minimum distance between the source and target node of the given transport plan
        min_dist = self.__compute_min_dist(transport_plan)

        # multiply the minimum distance with a random number sampled from a Frehet distribution with alpha=8
        return min_dist * np.random.weibull(8, 1)[0]
    

    def upgrade_edge(self, edge: tuple, weight: float=0.75) -> None:
        """
        Upgrades the given edge with the given weight.

        :param edge: An edge in the graph.
        :param weight: The weight of the edge.
        """
        # assert that edge is a tuple
        assert isinstance(edge, tuple)

        # assert that edge is in the graph
        assert edge in self.graph.edges, "Edge not in graph."

        # upgrade the given edge with the given weight
        self.graph.edges[edge]['weight'] = weight
        self.upgraded_edges.append(edge)


    def exact_solve(self, budget: int=6, preference: str='min_dist') -> None:
        """
        Solves the optimal transport problem exactly by computing the transport
        cost associated with all possible upgrades of budget edges.

        :param budget: The budget of the agent.
        :param preference: The preference of the agent.
        """
        if budget > 6:
            print("Warning: Exact solution may take a long time to compute.")
        
        # compute all possible subsets of edges of size budget
        subsets = self.__generate_subsets_tuples(list(self.graph.edges), budget)

        # create a dictionary that maps each subset of edges to the transport cost of the transport plans
        # if the edges in the subset were upgraded
        transport_costs = {}

        for subset in tqdm(subsets):
            # upgrade all edges in the subset
            for edge in subset:
                self.upgrade_edge(edge)

            # compute the transport cost of the transport plans if the edges in the subset were upgraded
            transport_costs[str(subset)] = self.compute_transport_cost(self.transport_plans, preference)

            # downgrade all edges in the subset
            for edge in subset:
                self.graph.edges[edge]['weight'] = 1
                self.upgraded_edges.remove(edge)

        # choose the subset of edges that minimizes the transport cost of the transport plans
        min_subset = min(transport_costs, key=transport_costs.get)

        # convert the subset from a string to a list of tuples
        min_subset = eval(min_subset)

        # upgrade all edges in the subset
        for edge in min_subset:
            self.upgrade_edge(edge)

        # compute the transport cost of the transport plans
        transport_cost = self.compute_transport_cost(self.transport_plans, preference)

        print("Transport Cost: ", transport_cost)
        

    def __generate_subsets_tuples(self, arr, k):
        def generate_subsets_helper(start, current_subset):
            if len(current_subset) == k:
                subsets.append(list(current_subset))
                return

            for i in range(start, len(arr)):
                current_subset.append(arr[i])
                generate_subsets_helper(i + 1, current_subset)
                current_subset.pop()

        if k < 0 or k > len(arr):
            return []

        subsets = []
        generate_subsets_helper(0, [])

        return subsets


    def greedy_solve(self, budget: int=25, preference: str='min_dist') -> None:
        for i in range(budget):
            chosen_edge = self.__choose_best_edge()
            self.upgrade_edge(chosen_edge)
            self.transport_costs.append(self.compute_transport_cost(self.transport_plans, preference))


    def __choose_best_edge(self):
        """
        Chooses the best edge to upgrade.

        :return: The best edge to upgrade.
        """
        # compute the transport cost of the current transport plans
        current_transport_cost = self.compute_transport_cost(self.transport_plans)

        # compute the transport cost of the transport plans if each edge in the graph was upgraded
        transport_costs = {}

        for edge in set(self.graph.edges):
            # upgrade the edge if it is not already upgraded
            if edge not in self.upgraded_edges:
                self.upgrade_edge(edge)

                # compute the transport cost of the transport plans if the edge was upgraded
                transport_costs[edge] = self.compute_transport_cost(self.transport_plans)

                # downgrade the edge and remove it from the list of upgraded edges
                self.graph.edges[edge]['weight'] = 1
                self.upgraded_edges.remove(edge)

        # choose the edge that minimizes the transport cost of the transport plans
        min_edge = min(transport_costs, key=transport_costs.get)

        return min_edge
    

    def mh_solve(self, budget: int=25, preference: str='min_dist') -> None:
        """
        Solves the optimal transport problem using the Metropolis-Hastings algorithm.

        :param budget: The budget of the agent.
        :param preference: The preference of the agent.
        """
        # randomly upgrade budget edges
        for i in range(budget):
            chosen_edge = self.__choose_random_edge()
            self.upgrade_edge(chosen_edge)

        initial_transport_cost = self.compute_transport_cost(self.transport_plans, preference)
        self.transport_costs.append(initial_transport_cost)

        transport_costs = [0, initial_transport_cost]

        # while the difference between the current transport cost and the previous transport cost is greater than 0.1
        # while self.__convergence_criterion(transport_costs):
            # beta = self.__mh_scheduler(len(transport_costs))
        for i in range(10000):
            beta = self.__mh_scheduler(i)
            # run a single Metropolis-Hastings step
            self.__mh_step(beta, transport_costs)
            print("Current Step: ", len(transport_costs) - 1)
            print("Current Transport Cost: ", transport_costs[-1])


    def __convergence_criterion(self, transport_costs: list) -> bool:
        """
        Depending on the length of the list of transport costs,
        if the last 10 transport costs have not had any improvement, return False.
        """
        for i in range(min(len(transport_costs), 10)):
            if transport_costs[-i] < transport_costs[-i-1]:
                return True            
        return False
    

    def __mh_scheduler(self, iter: int) -> float:
        """
        Returns the temperature parameter for the Metropolis-Hastings algorithm.

        :param iter: The current iteration of the algorithm.
        :return: The temperature parameter for the Metropolis-Hastings algorithm.
        """
        return np.exp(-0.0005 * iter)


    def __choose_random_edge(self, upgraded: bool=False) -> tuple:
        """
        Chooses a random edge from the graph.

        :param upgraded: Whether to choose a random upgraded edge or a random non-upgraded edge.
        :return: A random edge from the graph.
        """
        # choose a random upgraded edge
        if upgraded:
            return random.choice(self.upgraded_edges)
        # choose a random non-upgraded edge
        else:
            return random.choice(list(set(self.graph.edges) - set(self.upgraded_edges)))


    def __mh_step(self, beta: float, transport_costs: list) -> None:
        """
        Runs a single Metropolis-Hastings step. 
        The agent randomly upgrades an edge and decides whether to keep the upgrade or not.

        :param beta: The temperature parameter.
        :param transport_costs: A list of previous transport costs.
        """
        # choose a random upgraded edge to downgrade
        edge_to_remove = self.__choose_random_edge(upgraded=True)
        self.graph.edges[edge_to_remove]['weight'] = 1
        self.upgraded_edges.remove(edge_to_remove)

        # choose a random edge to upgrade
        chosen_edge = self.__choose_random_edge(upgraded=False)
        self.upgrade_edge(chosen_edge)

        # compute the transport cost of the transport plans if the edge was upgraded
        transport_cost = self.compute_transport_cost(self.transport_plans)

        # compute the acceptance probability
        acceptance_probability = min(1, np.exp(-beta * (transport_cost - transport_costs[-1])))

        # decide whether to keep the upgrade or not
        if np.random.uniform(0, 1) < acceptance_probability:
            transport_costs.append(transport_cost)
        else:
            self.graph.edges[chosen_edge]['weight'] = 1
            self.upgraded_edges.remove(chosen_edge)
            transport_costs.append(transport_costs[-1])

            # upgrade the edge that was downgraded
            self.graph.edges[edge_to_remove]['weight'] = 0.75
            self.upgraded_edges.append(edge_to_remove)


    def plot_graph(self):
        """
        Plots the graph and highlights the upgraded edges.
        """
        pos = {(x, y): (y, -x) for x, y in self.graph.nodes}

        # adjust the size of the network nodes based on their weight
        weights = [self.graph.nodes[node]['weight'] for node in self.graph.nodes]
        weights = [20 + weight**2 for weight in weights]

        # plot the network nodes with weight > 1 in blue, all other nodes in red
        nx.draw_networkx_nodes(self.graph, pos=pos, node_size=weights, node_color=['blue' if self.graph.nodes[node]['weight'] > 1 else 'grey' for node in self.graph.nodes])

        # plot the upgraded edges in red and increase their width by 2, 
        # all other edges in black
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=self.upgraded_edges, edge_color='red', width=2)
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=list(set(self.graph.edges) - set(self.upgraded_edges)), edge_color='black')
        
        plt.axis('off')
        plt.show()