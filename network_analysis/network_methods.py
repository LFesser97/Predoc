"""
network_methods.py

Created on Thu Mar 23 2023

@author: Lukas

This file contains all functions for analyzing homogeneous networks.
"""

# import packages

import networkx as nx
import numpy as np
import pandas as pd
import itertools


# functions for motif-based analysis of homogeneous networks

def count_motifs(graph: nx.Graph, motifs: dict) -> dict:
    """
    This function counts the motifs in a graph.
    Currently, only motifs with three nodes are supported.

    Parameters
    ----------
    graph : nx.Graph
        The graph.

    motifs : dict
        The motifs.

    Returns
    -------
    mcount : dict
        The motif counts.
    """
    mcount = dict(zip(motifs.keys(), list(map(int, np.zeros(len(motifs))))))
    nodes = graph.nodes()

    #We use iterools.product to have all combinations of three nodes in the
    #original graph. Then we filter combinations with non-unique nodes, because
    #the motifs do not account for self-consumption.

    triplets = list(itertools.product(*[nodes, nodes, nodes]))
    triplets = [trip for trip in triplets if len(list(set(trip))) == 3]
    triplets = map(list, map(np.sort, triplets))
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]

    #The for each each of the triplets, we (i) take its subgraph, and compare
    #it to all fo the possible motifs

    for trip in u_triplets:
        sub_gr = graph.subgraph(trip)
        mot_match = list(map(lambda mot_id: nx.is_isomorphic(sub_gr, motifs[mot_id]), motifs.keys()))
        match_keys = [list(motifs.keys())[i] for i in range(len(motifs)) if mot_match[i]]
        if len(match_keys) == 1:
            mcount[match_keys[0]] += 1

    return mcount


def compare_motif_frequency(network: nx.Graph, motifs: dict, num_baselines: int=100) -> dict:
    """
    This function compares the motif frequency of a graph to the motif frequency
    of a baseline graph.

    Parameters
    ----------
    network : nx.Graph
        The graph.

    motifs : dict
        The motifs.

    num_baselines : int, optional
        The number of baseline graphs to be generated. The default is 100.

    Returns
    -------
    motif_comparison : dict
        The motif comparison.
    """
    # Generate the baseline graphs from an Erdos-Renyi random graph with the same number of nodes and edges
    baseline_graphs = [nx.erdos_renyi_graph(
        network.number_of_nodes(), network.number_of_edges() / (network.number_of_nodes() * (network.number_of_nodes() - 1)))
        for i in range(num_baselines)]

    # Count the motifs in the baseline graphs
    baseline_motif_counts = [count_motifs(baseline_graphs[i], motifs) for i in range(num_baselines)]

    # for each motif in motifs, we calculate the mean and standard deviation in the baseline graphs
    baseline_motif_means = {motif: np.mean([baseline_motif_counts[i][motif]
                                            for i in range(num_baselines)]) for motif in motifs.keys()}

    baseline_motif_stds = {motif: np.std([baseline_motif_counts[i][motif]
                                          for i in range(num_baselines)]) for motif in motifs.keys()}

    # Count the motifs in the network
    try:
        network_motif_counts = network.graph["motif_counts"]

    except KeyError:
        print("Motifs in the network have not been counted yet. Counting motifs now...")
        network_motif_counts = count_motifs(network, motifs)

    # Calculate the z-scores for each motif
    motif_comparison = {motif: (network_motif_counts[motif] - baseline_motif_means[motif]) / baseline_motif_stds[motif]
                        for motif in motifs.keys()}
    
    return motif_comparison


def plot_significance_profile(network: nx.Graph, motifs: dict) -> None:
    """
    This function plots the significance profile of a network.

    Parameters
    ----------
    network : nx.Graph
        The graph.

    motifs : dict
        The motifs.

    num_baselines : int, optional
        The number of baseline graphs to be generated. The default is 100.

    Returns
    -------
    None.

    """
    pass