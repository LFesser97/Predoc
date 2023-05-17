"""
network_dataset_objects.py

Created on Tue Mar 14 2023

@author: Lukas

This file contains the classes for network datasets.
"""

# import packages

import networkx as nx
import pandas as pd
import os


# import other files from this project

import homogeneous_networks as hn
import heterogeneous_networks as ht


# define classes

class NetworkDataset:
    """
    This class contains all information about a network dataset.
    """
    def __init__(self):
        self.network_types = ["supplier", "shareholder"]
        self.dataset_types = ["csv", "json"]

        self.dataset = pd.DataFrame(columns=
                                    ["title", "company_type",
                                     "address", "est_date",
                                     "n_stocks_authorized", "stock_capital",
                                     "accounting_period", "dividend",
                                     "personnel", "main_shareholders",
                                     "n_employee", "banks",
                                     "locations", "suppliers"])

        self.networks = {}
        self.knowledge_graphs = {}

        self.known_knowledge_graph_motifs = {}


    # methods for reading in datasets

    def read_dataset_from_file(self, filename: str) -> None:
        """
        This method reads a network dataset from a file.

        Parameters
        ----------
        filename : The filename of the network dataset.

        Returns
        -------
        None : The dataset is stored in the dataset attribute.
        """
        assert filename.split(".")[-1] in self.dataset_types, "Unknown file extension."

        if filename.split(".")[-1] == "csv":
            self.dataset = ld.load_from_csv(filename, self.dataset)

        elif filename.split(".")[-1] == "json":
            self.dataset = ld.load_from_json(filename, self.dataset)


    # methods for homogeneous networks

    def create_homogeneous_network(self, network_type: str) -> None:
        """
        This method creates a specific network
        and appends it to the networks dictionary.

        Parameters
        ----------
        network_name : The name of the network.

        Returns
        -------
        None : The created network is appended to the networks dictionary.
        """
        assert network_type in self.network_types, "Unknown network type."

        self.networks[network_type] = hn.homogeneous_network(self.dataset, network_type)


    # methods for knowledge graphs

    def create_heterogeneous_network(self, network1_type: str, network2_type: str) -> None:
        """
        This method creates a heterogeneous network
        and appends it to the knowledge graph dictionary.

        Parameters
        ----------
        network1_type : The type of the first network.

        network2_type : The type of the second network.

        Returns
        -------
        None : The created knowledge graph is appended to the knowledge graph dictionary.
        """
        assert network1_type in self.network_types and network2_type in self.network_types, "Unknown network type."
        assert network1_type in self.networks, "The first network has not been created yet."
        assert network2_type in self.networks, "The second network has not been created yet."

        knowledge_graph = ht.connect_networks(self.networks[network1_type], self.networks[network2_type],
                                              network1_type, network2_type)

        self.knowledge_graphs[network1_type + "_" + network2_type] = knowledge_graph


    def extend_heterogeneous_network(self, ht_network_type: str, hm_network_type: str) -> None:
        """
        This method extends a heterogeneous network
        by  a homogeneous network and appends it to the knowledge graph dictionary.

        Parameters
        ----------
        ht_network_type : The type of the heterogeneous network.

        hm_network_type : The type of the homogeneous network.

        Returns
        -------
        None : The extended knowledge graph is appended to the knowledge graph dictionary.
        """
        assert ht_network_type in self.knowledge_graphs, "The heterogeneous network has not been created yet."
        assert hm_network_type in self.networks, "The homogeneous network has not been created yet."

        knowledge_graph = ht.extend_network(self.knowledge_graphs[ht_network_type],
                                            self.networks[hm_network_type],
                                            ht_network_type, hm_network_type)

        self.knowledge_graphs[ht_network_type + "_" + hm_network_type] = knowledge_graph