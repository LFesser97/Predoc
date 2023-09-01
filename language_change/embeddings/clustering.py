"""
clustering.py

Created on Wed Aug 30 2023

@Author: Lukas

This file contains all methods for clustering embeddings.
"""

# import packages
import argparse
import json
import pandas as pd
import numpy as np
import pickle
import scipy
import hdbscan
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from glob import glob


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
    def cluster_embeds(embedding_list: list, clustering_option:str = "slink") -> dict:
        """
        Cluster the embeddings in a list.
        """
        assert clustering_option in ["hdbscan", "slink", "hac"], "Invalid clustering option. Choose either 'hdbscan' , 'slink', or 'hac'."

        print("Number of embeddings:",len(embedding_list))
        embedding_array = np.array(embedding_list)

        if clustering_option == "hdbscan":
            clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size_hp, min_samples=min_samples_hp)
        elif clustering_option == "slink":
            clusterer = DBSCAN(eps=eps, min_samples=1, metric="cosine")
        else:
            clusterer = AgglomerativeClustering(distance_threshold=distance_threshold, affinity="cosine", linkage="single", n_clusters=None)

        clusterer.fit(embedding_array)

        clusters_dict = list(clusterer.labels_)

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


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Cluster embeddings.')
    parser.add_argument('input_file', type=str, help='The path to the embeddings.')
    parser.add_argument('output_file', type=str, help='The path to the output file.')
    parser.add_argument('clustering_option', type=str, help='The clustering option. Choose either "hdbscan", "slink", or "hac".')
    parser.add_argument('plot_clusters', type=bool, help='Whether to plot the clusters.')

    # hyperparameters for hdbscan, slink, and hac. Default values are cluster_size_hp = 1, min_samples_hp = 1, eps = 0.5, distance_threshold = 0.5
    parser.add_argument('--cluster_size_hp', type=int, help='The minimum size of clusters for hdbscan.')
    parser.add_argument('--min_samples_hp', type=int, help='The minimum number of samples in a neighborhood for hdbscan.')
    parser.add_argument('--eps', type=float, help='The maximum distance between two samples for slink.')
    parser.add_argument('--distance_threshold', type=float, help='The threshold to decide the cluster size for hac.')    
    args = parser.parse_args()

    # read the input file
    with open(args.input_file, 'rb') as f:
        embeddings_dict = pickle.load(f)

    # if hyperparameters are not specified, use default values
    cluster_size_hp = args.cluster_size_hp if args.cluster_size_hp else 1
    min_samples_hp = args.min_samples_hp if args.min_samples_hp else 1
    eps = args.eps if args.eps else 0.5    
    distance_threshold = args.distance_threshold if args.distance_threshold else 0.5

    # cluster embeddings
    clusters_dict = clustering.cluster_embeds(embeddings_dict, args.clustering_option)

    # write the output file
    with open(args.output_file, 'wb') as f:
        pickle.dump(clusters_dict, f)