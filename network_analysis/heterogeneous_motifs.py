"""
heterogeneous_motifs.py

Created on Tue May 16 2023

@author: Lukas

This file contains all functions for heterogenous motifs.
"""

# import packages

import networkx as nx
import numpy as np
import pandas as pd
import itertools


# class and methods for heterogeneous motifs

class heterogeneous_motif:
    """
    A class specifically for heterogeneous motifs.
    """
    def __init__(self, name: str, motif: nx.Graph) -> None:
        """
        This function initializes a heterogeneous motif.

        Parameters
        ----------
        name : The name of the motif.

        motif : The motif, i.e. a subgraph of a knowledge graph.

        Returns
        -------
        None : Initializes a heterogeneous motif.
        """
        assert nx.is_connected(motif), "The motif is not connected."
        assert all([motif.nodes[node]["type"] for node in motif.nodes()]), "The motif is not heterogeneous."


        self.name = name
        self.motif = motif
        self.frequency = -1
        self.relative_frequencies = {}


    def count_frequency(self, knowledge_graph: nx.Graph) -> None:
        """
        This function counts the frequency of a motif in a knowledge graph.
        Currently only supports motifs of size 3.

        Parameters
        ----------
        knowledge_graph : The knowledge graph.

        Returns
        -------
        None : Updates the frequency of the motif.
        """
        self.frequency = count_knowledge_motifs(self.motif, knowledge_graph)


# functions for motif-based analysis of knowledge graphs

def count_knowledge_motifs(graph: nx.Graph, motif: nx.Graph) -> int:
    """
    This function counts the motifs in a knowledge graph.

    Parameters
    ----------
    graph : A knowledge graph.

    motif : The motif.

    Returns
    -------
    mcount : The motif count.
    """
    # assert that the graph is a knowledge graph by checking whether nodes and edges have a type
    assert all([graph.nodes[node]["type"] for node in graph.nodes()])

    # treat the knowledge graph as a homogeneous network and get the candidate motifs
    m_candidates = get_candidate_motifs(graph, motif)

    # get the node and edge permutations of the input motif
    m_permutations = []

    m_permutations = get_motif_permutations(graph.subgraph(motif))

    # count the motif using test_candidate
    m_count = 0

    for candidate in m_candidates:
        if test_candidate(m_permutations[0], m_permutations[1], candidate):
            m_count += 1

    return m_count


def get_candidate_motifs(graph: nx.Graph, motif: nx.Graph) -> list:
    """
    This function extracts all candidate motifs from a homogeneous network.

    Parameters
    ----------
    graph : A homogeneous network.

    motif : The motif.

    Returns
    -------
    m_candidates : The candidates for the motif.
    """
    nodes = graph.nodes()

    triplets = list(itertools.product(*[nodes, nodes, nodes]))
    triplets = [trip for trip in triplets if len(list(set(trip))) == 3]
    triplets = map(list, map(np.sort, triplets))
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]

    m_candidates = []

    for trip in u_triplets:
        sub_gr = graph.subgraph(trip)
        if nx.is_isomorphic(sub_gr, motif):
            m_candidates.append(sub_gr)

    return m_candidates


def get_motif_permutations(motif: nx.Graph) -> tuple:
    """
    This function returns all permutations of a motif.

    Parameters
    ----------
    motif : A heterogeneous motif.

    Returns
    -------
    node_perms : The permutations of the node types.
    """
    # get a list of the types of the nodes in the motif
    node_types = [motif.nodes[node]['type'] for node in motif.nodes()]

    # get all permutations of the node types
    node_perms = list(itertools.permutations(node_types))

    # get a list of the types of the edges in the motif
    edge_types = [motif.edges[edge]['type'] for edge in motif.edges()]

    # get all permutations of the edge types
    edge_perms = list(itertools.permutations(edge_types))

    return node_perms, edge_perms


def test_candidate(node_perms: list, edge_perms: list, candidate: nx.Graph) -> bool:
    """
    This function tests whether a candidate is a motif.

    Parameters
    ----------
    node_perms : The permutations of the node types.

    edge_perms : The permutations of the edge types.

    candidate : A candidate motif.

    Returns
    -------
    is_motif : Whether the candidate is a motif.
    """
    # get a list of the types of the nodes in the candidate
    node_types = [candidate.nodes[node]['type'] for node in candidate.nodes()]

    # get a list of the types of the edges in the candidate
    edge_types = [candidate.edges[edge]['type'] for edge in candidate.edges()]

    # check whether the node and edge types match the permutations
    is_motif = (node_types in node_perms) and (edge_types in edge_perms)

    return is_motif