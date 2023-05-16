"""
term_sense_distances.py

Created on Mon May 15 2023

@autor: Lukas

This file contains all functions for calculating term-sense distances, as discussed in this
Confluence post: https://melissadell.atlassian.net/wiki/spaces/LC/pages/2706112535/How+different+are+the+usages+of+this+term
"""

# import packages

import pandas as pd
import numpy as np
import torch


# import other files from this project


# functions for calculating term-sense distances

def get_embedding_distance(text1_term_embeddings: torch.Tensor, text2_term_embeddings: torch.Tensor) -> float:
    """
    This function calculates the distance between two term embeddings.

    Parameters
    ----------
    text1_term_embeddings : The term embeddings of the first text.

    text2_term_embeddings : The term embeddings of the second text.

    Returns
    -------
    distance : The distance between the term embeddings.
    """
    assert text1_term_embeddings.shape == text2_term_embeddings.shape, "The term embeddings must have the same shape."

    # for every term embedded in the tensor, compute the angular distance between the two term embeddings
    distance = torch.zeros(text1_term_embeddings.shape[0])
    for i in range(text1_term_embeddings.shape[0]):
        distance[i] = get_angular_distance(text1_term_embeddings[i], text2_term_embeddings[i])

    # return the mean of the angular distances
    distance = torch.mean(distance)

    return distance


def get_angular_distance(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> float:
    """
    This function calculates the angular distance between two tensors.

    Parameters
    ----------
    tensor_1 : The first tensor.

    tensor_2 : The second tensor.

    Returns
    -------
    angular_distance : The arccos of the dot product of the two tensors, divided by pi.
    """
    assert tensor_1.shape == tensor_2.shape, "The tensors must have the same shape."

    angular_distance = torch.arccos(torch.dot(tensor_1, tensor_2))/(torch.pi)

    return angular_distance


def get_cluster_distance(text1_clusters: pd.DataFrame, text2_clusters: pd.DataFrame) -> float:
    """
    This function calculates the distance between two clusters.

    Parameters
    ----------
    text1_clusters : The cluster memberships of the term senses in the first text.

    text2_clusters : The clusters memberships of the term senses in the second text.

    Returns
    -------
    distance : The distance between the clusters.
    """
    assert text1_clusters.shape == text2_clusters.shape, "The clusters must have the same shape."

    # term, check if the cluster memberships are the same. If not, add 1 to the distance
    distance = 0
    for i in range(text1_clusters.shape[0]):
        if text1_clusters[i] != text2_clusters[i]:
            distance += 1

    # return the mean of the distances
    distance = distance/text1_clusters.shape[0]

    return distance


def get_non_embedding_distance(text1_terms: pd.DataFrame, text2_terms: pd.DataFrame, weights=False) -> float:
    """
    This function calculates the distance between two terms senses by
    comparing the frequencies of other terms used in texts 1 and 2.

    Parameters
    ----------
    text1_terms : The terms in the first text and their frequencies.

    text2_terms : The terms in the second text and their frequencies.

    Returns
    -------
    distance : The distance between the term senses.
    """
    assert text1_terms.shape == text2_terms.shape, "The terms must have the same shape."

    if weights:
        raise NotImplementedError("This weightes version of this function is not implemented yet.")
    
    # get the terms that are used in only one of the texts and add them to distance
    distance = 0
    for i in range(text1_terms.shape[0]):
        if text1_terms[i] == 0 and text2_terms[i] != 0:
            distance += 1
        elif text1_terms[i] != 0 and text2_terms[i] == 0:
            distance += 1

    # return the mean of the distances
    distance = distance/text1_terms.shape[0]

    return distance