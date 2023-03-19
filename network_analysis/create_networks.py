"""
create_networks.py

Created on Sun Mar 19 2023

@author: Lukas

This file contains all functions for creating networks.
"""

# import packages

import networkx as nx
import numpy as np
import pandas as pd


# functions for creating networks

def create_supplier_network(dataset: pd.DataFrame) -> nx.Graph:
    """
    This function creates a supplier network from a Japanese dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        The Japanese dataset.

    Returns
    -------
    network : nx.Graph
        The supplier network.
    """
    network = nx.DiGraph()

    try:
        for index, row in dataset.iterrows():
            for supplier in row["Suppliers"]:
                network.add_edge(supplier, row["Name"])

        # append the type of the nodes to the network
        for node in network.nodes():
            network.nodes[node]["type"] = "company"

    except KeyError:
        print("The dataset does not contain a 'Suppliers' column.")

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
            for shareholder in row["Shareholders"]:
                network.add_edge(shareholder, row["Name"])

        # append the type of the nodes to the network
        for node in network.nodes():
            if node in dataset["Name"].values:
                network.nodes[node]["type"] = "company"
            else:
                network.nodes[node]["type"] = "Bank/ Individual"

    except KeyError:
        print("The dataset does not contain a 'Shareholders' column.")

    return network