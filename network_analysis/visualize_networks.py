"""
visualize_networks.py

Created on Sun Mar 19 2023

@author: Lukas

This file contains all functions for visualizing networks.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt


# functions for visualizing networks

def visualize_supplier_network(network: nx.DiGraph) -> None:
    """
    This function visualizes a supplier network.

    Parameters
    ----------
    network : nx.DiGraph
        The supplier network.

    Returns
    -------
    None.
    """
    pos = nx.spring_layout(network)
    nx.draw(network, pos, with_labels=True, node_size=500, node_color="orange")
    labels = nx.get_edge_attributes(network, 'weight')
    nx.draw_networkx_edge_labels(network, pos, edge_labels=labels, label_pos=0.3, font_size=8)
    plt.show()


def visualize_shareholder_network(network: nx.Graph) -> None:
    """
    This function visualizes a shareholder network.

    Parameters
    ----------
    network : nx.Graph
        The shareholder network.

    Returns
    -------
    None.
    """
    pos = nx.spring_layout(network)

    # color the nodes according to their type
    node_colors = []

    for node in network.nodes():
        if network.nodes[node]["type"] == "company":
            node_colors.append("orange")
        elif network.nodes[node]["type"] == "Bank/ Individual":
            node_colors.append("lightblue")

    # draw the graph
    nx.draw(network, pos, with_labels=True, node_size=500, node_color=node_colors)
    labels = nx.get_edge_attributes(network, 'weight')
    nx.draw_networkx_edge_labels(network, pos, edge_labels=labels, label_pos=0.3, font_size=8)
    plt.show()


# functions for visualizing network statistics

def plot_significance_profile(significance_profile: dict) -> None:
    """
    This function plots a significance profile.

    Parameters
    ----------
    significance_profile : dict
        The significance profile.

    Returns
    -------
    None.
    """
    # normalize the significance profile
    significance_profile = {key: value / sum(significance_profile.values()) for key, value in significance_profile.items()}

    # plot the significance profile as a line plot with the x-axis being the motif and the y-axis being the significance
    plt.plot(list(significance_profile.keys()), list(significance_profile.values()))
    plt.xlabel("Motif")
    plt.ylabel("Normalized Z-Score")
    plt.title("NetworkSignificance Profile")
    plt.show()