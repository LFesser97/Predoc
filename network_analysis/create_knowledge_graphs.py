"""
knowledge_graphs.py

Created on Sun Mar 19 2023

@author: Lukas

This file contains all functions for creating and extending knowledge graphs.
"""

# import packages

import networkx as nx
import numpy as np
import pandas as pd


# functions for creating knowledge graphs

def connect_networks(network1: nx.Graph, network2: nx.Graph,
                     network1_type: str, network2_type: str) -> nx.Graph:
    """
    This function connects two networks to a knowledge graph.

    Parameters
    ----------
    network1 : nx.Graph
        The first network.

    network2 : nx.Graph
        The second network.

    network1_type : str
        The type of the first network.

    network2_type : str
        The type of the second network.

    Returns
    -------
    knowledge_graph : nx.Graph
        The knowledge graph.
    """
    knowledge_graph = nx.compose(network1, network2)

    # add the type of the edges to the knowledge graph
    for edge in knowledge_graph.edges():
        if edge[0] in network1.nodes():
            knowledge_graph.edges[edge]["type"] = network1_type
        else:
            knowledge_graph.edges[edge]["type"] = network2_type

    return knowledge_graph