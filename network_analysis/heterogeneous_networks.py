"""
heterogeneous_networks.py

Created on Sun Mar 19 2023

@author: Lukas

This file contains all functions for creating and extending knowledge graphs.
"""

# import packages

import networkx as nx
import numpy as np
import pandas as pd


# import other files from this project

import visualize_knowledge_graphs as vkg
import heterogeneous_motifs as ht


# functions for creating knowledge graphs

class heterogeneous_network(nx.Graph):
    """
    A class specifically for homogeneous networks that
    extends the networkx.Graph class with additional methods 
    for creating and visualizaing heterogeneous networks, 
    and for motif-based analysis.
    """
    def __init__(self, network1: nx.Graph, network2: nx.Graph,
                    network1_type: str, network2_type: str) -> None:
        """
        This function initializes a heterogeneous network.

        Parameters
        ----------
        network1 : The first network.

        network2 : The second network.

        network1_type : The type of the first network.

        network2_type : The type of the second network.

        Returns
        -------
        None : Initializes a heterogeneous network.
        """
        self = knowledge_graph = nx.compose(network1, network2)

        # add the type of the edges to the knowledge graph
        for edge in knowledge_graph.edges():
            if edge[0] in network1.nodes():
                knowledge_graph.edges[edge]["type"] = network1_type
            else:
                knowledge_graph.edges[edge]["type"] = network2_type


    def visualize(self) -> None:
        """
        This function visualizes a heterogeneous network.

        Parameters
        ----------
        None

        Returns
        -------
        None : Visualizes a heterogeneous network.
        """
        vkg.visualize_knowledge_graph(self)


    def create_motif(self, name: str, motif: nx.Graph) -> None:
        """
        This function creates a motif in a heterogeneous network.

        Parameters
        ----------
        name : The name of the motif.

        motif : The motif.

        Returns
        -------
        None : Creates a motif in a heterogeneous network.
        """
        self.motifs[name] = ht.motif(name, motif)