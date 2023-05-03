"""
knowledge_graph_methods.py

Created on Wed May 03 2023

author: Lukas

This file contains all functions for analyzing knowledge graphs.
"""

# import packages

import networkx as nx
import numpy as np
import pandas as pd
import itertools


# import other files from this project

import network_methods as nm


# functions for motif-based analysis of knowledge graphs

def count_knowledge_motifs(graph: nx.Graph, motifs: dict) -> dict:
    """
    This function counts the motifs in a knowledge graph.

    Parameters
    ----------
    graph : A knowledge graph.

    motifs : The motifs. 

    Returns
    -------
    mcount : The motif counts.
    """
    # assert that the graph is a knowledge graph by checking whether nodes and edges have a type
    assert all([graph.nodes[node]["type"] for node in graph.nodes()])

    # treat the knowledge graph as a homogeneous network and get the candidate motifs
    m_candidates = get_candidate_motifs(graph, motifs)

    # get the node and edge permutations of the input motifs
    m_permutations = {}

    for m_id in motifs.keys():
        m_permutations[m_id] = get_motif_permutations(graph.subgraph(motifs[m_id]))

    # count the motifs using test_candidate
    m_count = dict(zip(motifs.keys(), list(map(int, np.zeros(len(motifs))))))

    for m_id in m_candidates.keys():
        m_matches = [m_candidates[m_id][i] for i in range(len(m_candidates[m_id])) if
                        test_candidate(m_permutations[m_id][0], m_permutations[m_id][1], m_candidates[m_id][i])]
        m_count[m_id] = len(m_matches)

    return m_count


def get_candidate_motifs(graph: nx.Graph, motifs: dict) -> dict:
    """
    This function extracts all candidate motifs from a homogeneous network.

    Parameters
    ----------
    graph : A homogeneous network.

    motifs : The motifs.

    Returns
    -------
    m_candidates : The candidates for each motif.
    """
    m_candidates = dict(zip(motifs.keys(), [[] for i in range(len(motifs.keys()))]))
    nodes = graph.nodes()

    triplets = list(itertools.product(*[nodes, nodes, nodes]))
    triplets = [trip for trip in triplets if len(list(set(trip))) == 3]
    triplets = map(list, map(np.sort, triplets))
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]

    for trip in u_triplets:
        sub_gr = graph.subgraph(trip)
        mot_match = list(map(lambda mot_id: nx.is_isomorphic(sub_gr, motifs[mot_id]), motifs.keys()))
        match_keys = [list(motifs.keys())[i] for i in range(len(motifs)) if mot_match[i]]

        m_candidates[match_keys[0]].append(trip)

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