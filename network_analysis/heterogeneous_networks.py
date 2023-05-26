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
import homogeneous_networks as hn


# functions for creating knowledge graphs

class heterogeneous_network():
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
        # create the knowledge graph by combining the two networks
        self.graph = nx.compose(network1, network2)

        # check if network1 and network2 are both instances of homogeneous_network
        if isinstance(network1, hn.homogeneous_network) and isinstance(network2, hn.homogeneous_network):

            # add the type of the edges to the knowledge graph
            for edge in self.graph.edges():
                if edge[0] in network1.nodes():
                    self.graph.edges[edge]["type"] = network1_type
                else:
                    self.graph.edges[edge]["type"] = network2_type

        # otherwise, network1 is a heterogeneous network, which we extend
        else:
            for edge in self.graph.edges():
                if edge[0] in network2.nodes():
                    self.graph.edges[edge]["type"] = network1_type


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
        vkg.visualize_knowledge_graph(self.graph)


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