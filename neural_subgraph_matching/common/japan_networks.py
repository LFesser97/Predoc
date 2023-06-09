# import packages

import numpy as np
import pandas as pd
import networkx as nx


# define all node types and edge types in a Japanese network

node_types = ['individual', 'bank', 'company']
edge_types = ['shareholder', 'board member', 'supplier', 'employee', 'bank loan']


# function to create a random network based on the node and edge types

def create_japan_graph(node_types: list = node_types, edge_types: list = edge_types, n_nodes: list = [100, 10, 10], 
                       probabilities: dict = {'shareholder': 0.5,'board member': 0.5, 'supplier': 0.5, 'employee': 0.5, 'bank loan': 0.5}) -> nx.Graph: 
    """
    Create a random network based on the node and edge types.
    
    Parameters
    ----------
    node_types : List of node types.
    edge_types : List of edge types.
    n_nodes : List with number of nodes for each node type.
    probabilities : Dictionary with probabilities for each edge type.
        
    Returns
    -------
    G : A random network.
    """
    
    # create a random network
    G = nx.Graph()
    
    # add nodes and node types
    for i, node_type in enumerate(node_types):
        G.add_nodes_from(range(sum(n_nodes[:i]), sum(n_nodes[:i+1])), node_type=node_type)
    
    # add edges and edge types
    for i, edge_type in enumerate(edge_types):
        if edge_type == 'shareholder':
            for node in G.nodes():
                if G.nodes[node]['node_type'] == 'individual' or G.nodes[node]['node_type'] == 'bank':
                    for node_2 in G.nodes():
                        if G.nodes[node_2]['node_type'] == 'company':
                            if np.random.uniform() < probabilities[edge_type]:
                                G.add_edge(node, node_2, edge_type=edge_type)

        elif edge_type == 'board member':
            for node in G.nodes():
                if G.nodes[node]['node_type'] == 'individual':
                    for node_2 in G.nodes():
                        if G.nodes[node_2]['node_type'] == 'company':
                            if np.random.uniform() < probabilities[edge_type]:
                                G.add_edge(node, node_2, edge_type=edge_type)

        elif edge_type == 'supplier':
            for node in G.nodes():
                if G.nodes[node]['node_type'] == 'company':
                    for node_2 in G.nodes():
                        if G.nodes[node_2]['node_type'] == 'company':
                            if np.random.uniform() < probabilities[edge_type]:
                                G.add_edge(node, node_2, edge_type=edge_type)

        elif edge_type == 'employee':
            for node in G.nodes():
                if G.nodes[node]['node_type'] == 'individual':
                    for node_2 in G.nodes():
                        if G.nodes[node_2]['node_type'] == 'company':
                            if np.random.uniform() < probabilities[edge_type]:
                                G.add_edge(node, node_2, edge_type=edge_type)
    
    return G