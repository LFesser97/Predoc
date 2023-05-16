"""
homogeneous_motifs.py

Created on Thu May 11 2023

@author: Lukas

This file contains all functions for homogeneous motifs.
"""

# import packages

import networkx as nx
import numpy as np
import pandas as pd
import itertools


# class and methods for homogeneous motifs

class homogeneous_motif:
    """
    A class specifically for homogeneous motifs.
    """
    def __init__(self, name: str, motif: nx.Graph) -> None:
        """
        This function initializes a homogeneous motif.

        Parameters
        ----------
        name : The name of the motif.

        motif : The motif.

        Returns
        -------
        None : Initializes a homogeneous motif.
        """
        self.name = name
        self.motif = motif
        self.frequency = -1
        self.relative_frequencies = {}


    def count_frequency(self, network: nx.Graph) -> None:
        """
        This function counts the frequency of a motif in a network.
        Currently only supports motifs of size 3.

        Parameters
        ----------
        network : The underlying network.

        Returns
        -------
        None : Updates the frequency of the motif.
        """
        self.frequency = count_frequency(self.motif, network)


    def count_relative_frequency(self, network: nx.Graph, baselines: list, baseline_name: str, num_baselines: int) -> None:
        """
        This function counts the relative frequency of a motif in a network.

        Parameters
        ----------
        network : The underlying network.

        baselines : The baseline networks.

        baseline_name : The name of the baseline network.

        num_baselines : The number of baseline networks.

        Returns
        -------
        None : Updates the relative frequency of the motif.
        """
        # Count the motifs in the baseline graphs
        baseline_motif_counts = [count_frequency(self, baselines[i]) for i in range(num_baselines)]

        # Calculate the mean and standard deviation in the baseline graphs
        baseline_motif_mean = np.mean(baseline_motif_counts)
        baseline_motif_std = np.std(baseline_motif_counts)

        # Count the motifs in the network
        if self.frequency == -1:
            print("Motifs in the network have not been counted yet. Counting motifs now...")
            self.count_frequency(network)
            network_motif_count = self.frequency

        else:
            network_motif_count = self.frequency

        # Calculate the z-score for the motif
        z_score = (network_motif_count - baseline_motif_mean) / baseline_motif_std

        self.relative_frequencies[baseline_name] = z_score


def count_frequency(motif: nx.Graph, network: nx.Graph) -> int:
    """
    This function counts the frequency of a motif in a network.
    Currently only supports motifs of size 3.

    Parameters
    ----------
    motif : The motif.

    network : The underlying network.

    Returns
    -------
    frequency : The frequency of the motif.
    """
    nodes = network.nodes()

    # We use iterools.product to have all combinations of three nodes in the
    # original graph. Then we filter combinations with non-unique nodes, because
    # the motifs do not account for self-consumption.

    triplets = list(itertools.product(*[nodes, nodes, nodes]))
    triplets = [trip for trip in triplets if len(list(set(trip))) == 3]
    triplets = map(list, map(np.sort, triplets))
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]

    # The for each each of the triplets, we (i) take its subgraph, and compare
    # it to all fo the possible motifs

    frequency = 0

    for trip in u_triplets:
        sub_gr = network.subgraph(trip)
        if nx.is_isomorphic(sub_gr, motif):
            frequency += 1

    return frequency