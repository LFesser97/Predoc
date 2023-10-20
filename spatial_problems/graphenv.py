"""
graphenv.py

Created on Fri Oct 20 2023

@author: Lukas

This file contains all code for setting up the optimal transport problem.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

        # sample a point from a 2D normal distribution with mean_1 and var_1
        for i in range(num_plans):
            source = np.random.multivariate_normal(mean_1, var_1)
            target = np.random.multivariate_normal(mean_2, var_2)

            # convert the sampled points to nodes in the graph
            source_node = (int(source[0]), int(source[1]))
            target_node = (int(target[0]), int(target[1]))

            print(source_node, target_node)

            # append the transport plan to the list of transport plans
            plans.append((source_node, target_node)) 

        self.transport_plans = plans

        # set the node weights of the graph to be the number of transport plans that start or end at the node
        for node in self.graph.nodes:
            self.graph.nodes[node]['weight'] = 0

        for plan in self.transport_plans:
            source_node = plan[0]
            target_node = plan[1]
            self.graph.nodes[source_node]['weight'] += 1
            self.graph.nodes[target_node]['weight'] += 1
    

    def compute_transport_cost(self, transport_plans: list, preference: str='min_dist') -> float:
        """
        Computes the transport cost of the given transport plans.

        :param transport_plans: A list of transport plans.
        :param preference: The preference of the agent.
        :return: The transport cost of the given transport plans.
        """
        # assert that preference is either 'min_dist' or 'min_cost'
        assert preference in ['min_dist'], "Preference not supported. Preference must be 'min_dist'."

        # compute the transport cost of the given transport plans
        transport_cost = 0
        for plan in transport_plans:
            if preference == 'min_dist':
                transport_cost += self.__compute_min_dist(plan)
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
        # compute the minimum distance between the source and target node,
        # an edge that has been upgraded has a weight of 0.75, an edge that has not been upgraded has a weight of 1
        min_dist = sum([self.graph.edges[edge]['weight'] for edge in zip(shortest_path[:-1], shortest_path[1:])])

        return min_dist
    

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


    def plot_graph(self):
        """
        Plots the graph and highlights the upgraded edges.
        """
        pos = {(x, y): (y, -x) for x, y in self.graph.nodes}

        # adjust the size of the network nodes based on their weight
        weights = [self.graph.nodes[node]['weight'] for node in self.graph.nodes]
        weights = [50 + weight**2 for weight in weights]
        nx.draw_networkx_nodes(self.graph, pos=pos, node_size=weights, node_color='blue')

        # plot the upgraded edges in red, all other edges in black
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=self.upgraded_edges, edge_color='red')
        nx.draw_networkx_edges(self.graph, pos=pos, edgelist=list(set(self.graph.edges) - set(self.upgraded_edges)), edge_color='black')
        
        plt.axis('off')
        plt.show()