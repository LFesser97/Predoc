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
            for supplier in row["suppliers"]:
                network.add_edge(supplier, row["title"])

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
            for shareholder in row["main_shareholders"]:
                network.add_edge(shareholder, row["title"])

        # append the type of the nodes to the network
        for node in network.nodes():
            if node in dataset["title"].values:
                network.nodes[node]["type"] = "company"
            else:
                network.nodes[node]["type"] = "Bank/ Individual"

    except KeyError:
        # print("The dataset does not contain a 'Shareholders' column.")
        for index, row in dataset.iterrows():
            for shareholder in row["shareholders"]:
                network.add_edge(shareholder, row["title"])

        # append the type of the nodes to the network
        for node in network.nodes():
            if node in dataset["title"].values:
                network.nodes[node]["type"] = "company"
            else:
                network.nodes[node]["type"] = "Bank/ Individual"

    return network