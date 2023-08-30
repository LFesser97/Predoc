"""
headlines.py

Created on Tue Aug 22 2023

@author: Lukas

This file contains all the 
"""

# import packages

import nltk
nltk.download('words')
from nltk.corpus import words

english_words = set(words.words())


# auxiliary functions

class dictionary_method:
    """
    A class with static methods for pulling content words from headlines.
    """
    @staticmethod
    def filter_non_dictionary_words(input_words: list) -> list:
        """
        Given a list of words, return a list of words that are in the
        English dictionary.
        """
        return [word for word in input_words if word in english_words]
    

    @staticmethod
    def get_headline_indices(target_words: list, headlines: dict) -> dict:
        """
        Given a list of target words and a dictionary of headlines,
        return a dictionary with the target words as keys and a dictionary
        with years as keys and indices of headlines containing the target
        word as values.
        """
        headline_indices = {}

        for word in target_words:
            word_indices = {}
            for decade in headlines.keys():
                word_indices[decade] = [i for i in range(len(headlines[decade])) if word in headlines[decade][i]]
            headline_indices[word] = word_indices

        return headline_indices
    

    @staticmethod
    def get_headlines_from_indices(indices: dict, embeddings: dict, headlines: dict) -> dict:
        """
        Given alist with target words and a dictionary
        with years as keys and indices of headlines containing the target
        word as values, return a dictionary with the target words as keys
        and a dictionary with years as keys and headlines containing the
        target word as values.
        """
        target_headlines = {}

        for word in indices.keys():
            word_headlines = {}
            for decade in indices[word].keys():
                word_headlines[decade] = [[headlines[decade][i], embeddings[decade][i], i] for i in indices[word][decade]]
            target_headlines[word] = word_headlines

        return target_headlines


    @staticmethod
    def get_nearest_neighbors(target_headlines: dict, definitions: dict) -> dict:
        """
        Given a dictionary with embeddings under target_headlines[word][decade][i][1]
        and a dictionary with a list of embeddings for definitions under
        definitions[word], return an updated target_headlines dictionary
        with the nearest neighbors of the target words under
        target_headlines[word][decade][i][3].
        """
        for word in target_headlines.keys():
            for decade in target_headlines[word].keys():
                for i in range(len(target_headlines[word][decade])):
                    target_headlines[word][decade][i].append(headlines.get_nearest_neighbor(target_headlines[word][decade][i][1], definitions[word]))

        return target_headlines


    @staticmethod
    def get_nearest_neighbor(embedding: list, definitions: list) -> list:
        """
        Given a list of embeddings and a list of definitions, return the
        definition with the nearest embedding.
        """
        distances = []
        for definition in definitions:
            distances.append(headlines.get_distance(embedding, definition))
        return definitions[distances.index(min(distances))]
    

    @staticmethod
    def get_cosine_distance(embedding1: list, embedding2: list) -> float:
        """
        Given two embeddings, return the cosine distance between them.
        """
        return 1 - headlines.get_cosine_similarity(embedding1, embedding2)
    

    @staticmethod
    def get_cosine_similarity(embedding1: list, embedding2: list) -> float:
        """
        Given two embeddings, return the cosine similarity between them.
        """
        return sum([embedding1[i] * embedding2[i] for i in range(len(embedding1))]) / (headlines.get_norm(embedding1) * headlines.get_norm(embedding2))
    

    @staticmethod
    def get_norm(embedding: list) -> float:
        """
        Given an embedding, return its norm.
        """
        return sum([embedding[i] ** 2 for i in range(len(embedding))]) ** 0.5
    

    @staticmethod
    def create_usage_histogram(target_headlines: dict) -> None:
        """
        Given a dictionary with cluster id under
        nn_dict[word][decade][i][3] for document i, create a usage
        histogram for each word.
        """
        for word in target_headlines.keys():
            for decade in target_headlines[word].keys():
                cluster_ids = [target_headlines[word][decade][i][3] for i in range(len(target_headlines[word][decade]))]

                # create a dictionary with cluster ids as keys and counts as values
                cluster_id_counts = {}
                for cluster_id in cluster_ids:
                    if cluster_id not in cluster_id_counts:
                        cluster_id_counts[cluster_id] = 1
                    else:
                        cluster_id_counts[cluster_id] += 1

        return cluster_id_counts
    

# class for headline clustering

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy
import hdbscan

###Import clustering libraries
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from glob import glob
import plotly.express as px
import plotly.graph_objects as go

import json


class clustering:
    @staticmethod
    def aggregate_decades(headlines_dict: dict, target_words: list) -> pd.DataFrame:
        """
        Aggregate the dataframes created by make_df_with_clusters() for all decades
        and for all target words in a list.
        """
        df_list = []

        decades = list(headlines_dict[target_words[0]].keys())

        for word in target_words:
            all_headlines = [[headlines_dict[word][decade][i][0], headlines_dict[word][decade][i][1], decade]
                             for decade in decades for i in range(len(headlines_dict[word][decade]))]

            embedding_list = [all_headlines[i][1] for i in range(len(all_headlines))]
            clusters = clustering.cluster_embeds(embedding_list)

            for decade in decades:
                df_list.append(clustering.make_df_with_clusters([all_headlines[i][0] for i in range(len(all_headlines)) if all_headlines[i][2] == decade],
                                                                [clusters[i] for i in range(len(clusters)) if all_headlines[i][2] == decade],
                                                                decade))

        df = pd.concat(df_list)

        return df
        

    @staticmethod
    def make_df_with_clusters(headlines: list, clusters: list, decade: str) -> pd.DataFrame:
        """
        Create a DataFrame with the cluster labels and the decade and text.
        """
        df = pd.DataFrame({"headline": headlines, "cluster": clusters, "decade": decade})

        return df


    @staticmethod
    def cluster_embeds(embedding_list: list, clustering_option:str = "hdbscan") -> dict:
        """
        Cluster the embeddings in a list.
        """
        assert clustering_option in ["hdbscan", "slink", "hac"], "Invalid clustering option. Choose either 'hdbscan' , 'slink', or 'hac'."

        print("Number of embeddings:",len(embedding_list))
        embedding_array = np.array(embedding_list)

        if clustering_option == "hdbscan":
            clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size_hp, min_samples=min_samples_hp)
        elif clustering_option == "slink":
            clusterer = DBSCAN(eps=0.5, min_samples=1, metric="cosine")
        else:
            clusterer = AgglomerativeClustering(distance_threshold=distance_threshold, affinity="cosine", linkage="single", n_clusters=None)

        clusterer.fit(embedding_array)

        clusters_dict = list(clusterer.labels_)
        # print(max(clusterer.labels_))
        print("Unique Cluster IDs: ", np.unique(clusterer.labels_, return_counts=True))

        return clusters_dict


    @staticmethod
    def plot_clusters(df: pd.DataFrame) -> None:
        """
        Plot clusters with texts.
        """
        # Remove the noise cluster
        df = df[df["cluster"] != -1]

        fig = px.scatter(df, x='decade', y='cluster', custom_data=['headline'],color='cluster')

        # Customize the hover text to show the wrapped text
        fig.update_traces(hovertemplate="<br>".join([
            "Year: %{x}",
            "Cluster: %{y}",
            "Text: <br>%{customdata[0]}",
        ]))

        # Add axis labels
        fig.update_layout(
            title='Scatter Plot of Clusters by Year',
            xaxis_title='Year',
            yaxis_title='Cluster ID',
        )

        # Show the plot
        fig.show()