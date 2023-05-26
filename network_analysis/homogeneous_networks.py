"""
homogeneous_networks.py

Created on Sun Mar 19 2023

@author: Lukas

This file contains all functions for homogeneous networks.
"""

# import packages

import networkx as nx
import numpy as np
import pandas as pd


# import other files from this project

import homogeneous_motifs as hm
import visualize_networks as vn


# functions for creating networks

class homogeneous_network():
    """
    A class specifically for homogeneous networks that
    extends the networkx.Graph class with additional methods,
    primarily for motif-based analysis.
    """
    def __init__(self, dataset: pd.DataFrame, network_type: str) -> None:
        """
        This function initializes a homogeneous network.

        Parameters
        ----------
        dataset : the dataset from which the network is created.

        network_type : the type of the network.

        Returns
        -------
        None : Initializes a homogeneous network.
        """
        assert network_type in ["supplier", "shareholder"], "The method for creating this type of network has not been implemented yet."

        if network_type == "supplier":
            self.graph = create_supplier_network(dataset)

        elif network_type == "shareholder":
            self.graph = create_shareholder_network(dataset)

        self.network_type = network_type
        self.motifs = {} # dictionary with names as keys and motifs as values
        self.baselines = {} # dictionary with names as keys and lists of baselines as values


    def create_motif(self, name: str, motif: nx.Graph) -> None:
        """
        This function creates a motif.

        Parameters
        ----------
        name : The name of the motif.

        motif : The motif.

        Returns
        -------
        None : Creates a motif.
        """
        self.motifs[name] = hm.homogeneous_motif(name, motif)


    def count_motif_frequency(self, motif_name: str) -> None:
        """
        This function counts the frequency of a motif in a network.

        Parameters
        ----------
        motif_name : The name of the motif.

        Returns
        -------
        None : Appends the frequency to the motif object.
        """
        motif = self.motifs[motif_name]
        motif.count_frequency(self.graph) # note that self here refers to the homogeneous network oject


    def count_motif_relative_frequency(self, motif_name: str, baseline_name: str, num_baselines: int) -> None:
        """
        This function counts the relative frequency of a motif in a network.

        Parameters
        ----------
        motif_name : The name of the motif.

        baseline_name : The name of the baseline.

        num_baselines : The number of baselines.

        Returns
        -------
        None : Appends the relative frequency to the motif object.
        """
        assert baseline_name in self.baselines.keys(), "The baseline does not exist."
        assert 0 < num_baselines <= len(self.baselines[baseline_name]), "The number of baselines is not valid."

        motif = self.motifs[motif_name]
        motif.count_relative_frequency(self.graph, self.baselines[baseline_name], baseline_name, num_baselines)


    def create_baselines(self, name: str, num_baselines: int) -> None:
        """
        This function creates baselines for a network.

        Parameters
        ----------
        name : The name of the baseline.

        num_baselines : The number of baselines.

        Returns
        -------
        None : Creates baselines for a network.
        """
        assert name not in self.baselines.keys(), "The baseline already exists."
        assert name in ["erdos_renyi"], "Unknown baseline type."

        if name == "erdos_renyi":
            self.baselines[name] = [create_erdos_renyi_baseline(self.graph) for i in range(num_baselines)]


    def plot_significance_profile(self, motif_names: str, baseline_name: str) -> None:
        """
        This function plots the significance profile of a list of motifs.

        Parameters
        ----------
        motif_names : The names of the motifs.

        baseline_name : The name of the baseline.

        Returns
        -------
        None : Plots the significance profile.
        """
        assert baseline_name in self.baselines.keys(), "The baseline does not exist."

        for motif_name in motif_names:
            assert motif_name in self.motifs.keys(), "The motif does not exist."

        significance_profile = {motif_name : self.motifs[motif_name].relative_frequency for motif_name in motif_names}

        vn.plot_significance_profile(significance_profile, baseline_name)


    def visualize_network(self) -> None:
        """
        This function visualizes a the network.

        Parameters
        ----------
        None

        Returns
        -------
        None : Plots the network.
        """
        if self.network_type == "supplier":
            vn.visualize_supplier_network(self.graph)

        elif self.network_type == "shareholder":
            vn.visualize_shareholder_network(self.graph)

        else:
            print("Unknown network type. Visualization method not implemented yet.")
        

# methods for creating homogeneous networks
    
def create_supplier_network(dataset: pd.DataFrame) -> nx.Graph:
    """
    This function creates a supplier network from a Network dataset.

    Parameters
    ----------
    dataset : The Japanese dataset.

    Returns
    -------
    network : The supplier network.
    """
    network = nx.DiGraph()

    for index, row in dataset.iterrows():
        for supplier in row["suppliers"]:
            network.add_edge(supplier, row["name"])

    # append the type of the nodes to the network
    for node in network.nodes():
        network.nodes[node]["type"] = "company"

    return network


def create_shareholder_network(dataset: pd.DataFrame) -> nx.Graph:
    """
    This function creates a shareholder network from a Japanese dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        The Japanese dataset.

    Returns
    -------
    network : nx.Graph
        The shareholder network.
    """
    network = nx.DiGraph()

    try:
        for index, row in dataset.iterrows():
            for shareholder in row["shareholders"]:
                network.add_edge(shareholder, row["name"])

        # append the type of the nodes to the network
        for node in network.nodes():
            if node in dataset["name"].values:
                network.nodes[node]["type"] = "company"
            else:
                network.nodes[node]["type"] = "Bank/ Individual"

    except KeyError:
        # print("The dataset does not contain a 'Shareholders' column.")
        for index, row in dataset.iterrows():
            for shareholder in row["shareholders"]:
                network.add_edge(shareholder, row["name"])

        # append the type of the nodes to the network
        for node in network.nodes():
            if node in dataset["name"].values:
                network.nodes[node]["type"] = "company"
            else:
                network.nodes[node]["type"] = "Bank/ Individual"

    return network


# methods for creating baselines

def create_erdos_renyi_baseline(network: nx.Graph) -> nx.Graph:
    """
    This function creates an Erdos-Renyi baseline for a network.

    Parameters
    ----------
    network : The network.

    Returns
    -------
    baseline : The baseline.
    """
    baseline = nx.Graph()

    # create a random graph with the same number of nodes and edges as the network
    baseline.add_nodes_from(network.nodes())
    baseline.add_edges_from(np.random.choice(list(network.nodes()), size=network.number_of_edges(), replace=True))

    return baseline