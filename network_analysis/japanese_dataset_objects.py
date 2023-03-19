"""
japanese_dataset_objects.py

Created on Tue Mar 14 2023

@author: Lukas

This file contains the classes for the Japanese datasets.
"""

# import packages

import networkx as nx


# import other files from this project

import create_networks as cn
import visualize_networks as vn
import create_knowledge_graphs as kg
import visualize_knowledge_graphs as vkg


# define classes

class JapaneseDataset:
    """
    This class contains all information about a Japanese dataset.
    """
    def __init(self):
        self.dataset = None
        self.networks = {}
        self.knowledge_graphs = {}


    def read_dataset_from_file(self, filename):
        """
        This method reads a Japanese dataset from a file.

        Parameters
        ----------
        filename : str
            The filename of the Japanese dataset.

        Returns
        -------
        None.
        """
        pass


    def create_network(self, network_type: str) -> None:
        """
        This method creates a specific network
        and appends it to the networks dictionary.

        Parameters
        ----------
        network_name : str
            The name of the network.

        Returns
        -------
        None.
            The created network is appended to the networks dictionary.
        """
        assert network_type in ["supplier", "shareholder"], "Unknown network type."

        if network_type == "supplier":
            network = cn.create_supplier_network(self.dataset)

        elif network_type == "shareholder":
            network = cn.create_shareholder_network(self.dataset)

        self.networks[network_type] = network


    def write_network_to_gml(self, network_type, filename):
        """
        This method writes a network to a gml file.

        Parameters
        ----------
        network_name : str
            The name of the network.

        filename : str
            The filename of the gml file.

        Returns
        -------
        None.
        """
        pass


    def visualize_network(self, network_type: str) -> None:
        """
        This method visualizes a network.

        Parameters
        ----------
        network_name : str
            The name of the network.

        Returns
        -------
        None.
            Plots the network.
        """
        assert network_type in ["supplier", "shareholder"], "Unknown network type."

        if network_type == "supplier":
            assert network_type in self.networks, "The supplier network has not been created yet."
            vn.visualize_supplier_network(self.networks[network_type])

        elif network_type == "shareholder":
            assert network_type in self.networks, "The shareholder network has not been created yet."
            vn.visualize_shareholder_network(self.networks[network_type])


    def visualize_knowledge_graph(self, knowledge_graph: str) -> None:
        """
        This method visualizes a knowledge graph.

        Parameters
        ----------
        knowledge_graph : str
            The name of the knowledge graph.

        Returns
        -------
        None.
            Plots the knowledge graph.
        """
        assert knowledge_graph in self.knowledge_graphs, "The knowledge graph has not been created yet."

        vkg.visualize_knowledge_graph(self.knowledge_graphs[knowledge_graph])


    def connect_networks(self, network_type1: str, network_type2: str) -> None: 
        """
        This method connects two networks into a knowledge graph.

        Parameters
        ----------
        network_name1 : str
            The name of the first network.
        network_name2 : str
            The name of the second network.

        Returns
        -------
        None.
            The created knowledge graph is appended to the knowledge graphs dictionary.         
        """
        assert network_type1 in ["supplier", "shareholder"], "Unknown network type."
        assert network_type2 in ["supplier", "shareholder"], "Unknown network type."

        assert network_type1 in self.networks, "The first network has not been created yet."
        assert network_type2 in self.networks, "The second network has not been created yet."

        knowledge_graph = kg.connect_networks(self.networks[network_type1], self.networks[network_type2],
                                              network_type1, network_type2)

        self.knowledge_graphs[network_type1 + "_" + network_type2] = knowledge_graph


    def extend_knowledge_graph(self, knowledge_graph: str, network_type: str) -> None:
        """
        This method extends a knowledge graph with a network.

        Parameters
        ----------
        knowledge_graph : str
            The name of the knowledge graph to be extended.

        network_name : str
            The name of the network.

        Returns
        -------
        None.
            The extended knowledge graph is appended to the knowledge graphs dictionary.
        """
        pass


    def connect_knowledge_graphs(self, knowledge_graph1: str, knowledge_graph2: str) -> None:
        """
        This method connects two knowledge graphs.

        Parameters
        ----------
        knowledge_graph1 : str
            The name of the first knowledge graph.

        knowledge_graph2 : str
            The name of the second knowledge graph.

        Returns
        -------
        None.
            The created knowledge graph is appended to the knowledge graphs dictionary.
        """
        pass