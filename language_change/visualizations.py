"""
visualizations.py

Created on Tue Aug 08 2023

@author: Lukas

This file contains all methods for visualizing the results of the
cluster analysis.
"""

# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# given a dataframe with columns 'year' and 'cluster', create a dictionary
# of the form {year: {cluster: relative frequency}}
def get_cluster_frequencies(df: pd.DataFrame) -> dict:
    """
    Given a dataframe with columns 'year' and 'cluster', create a
    dictionary of the form {year: {cluster: relative frequency}}.

    Parameters
    ----------
    df : A dataframe with columns 'year' and 'cluster'.

    Returns
    ----------
    A dictionary of the form {year: {cluster: relative frequency}}.
    """
    # get the number of documents in each year
    year_counts = df['year'].value_counts()

    # create a dictionary of the form {year: {cluster: relative frequency}}
    cluster_frequencies = {}
    for year in df['year'].unique():
        # get the number of documents in the year
        year_count = year_counts[year]

        # get the number of documents in each cluster in the year
        cluster_counts = df[df['year'] == year]['cluster'].value_counts()

        # create a dictionary of the form {cluster: relative frequency}
        cluster_frequencies[year] = {}
        for cluster in df['cluster'].unique():
            # get the number of documents in the cluster
            cluster_count = cluster_counts[cluster]

            # calculate the relative frequency of the cluster in the year
            cluster_frequencies[year][cluster] = cluster_count / year_count

    return cluster_frequencies


def aggregate_to_decades(cluster_frequencies: dict) -> dict:
    """
    Given a dictionary of the form {year: {cluster: relative frequency}},
    aggregate the relative frequencies to decades.

    Parameters
    ----------
    cluster_frequencies : A dictionary of the form {year: {cluster: relative frequency}}.

    Returns
    ----------
    A dictionary of the form {decade: {cluster: relative frequency}}.
    """
    # create a dictionary of the form {decade: {cluster: relative frequency}}
    decade_frequencies = {}
    for year in cluster_frequencies:
        # get the decade of the year
        decade = year - (year % 10)

        # add the cluster frequencies to the decade frequencies
        if decade in decade_frequencies:
            for cluster in cluster_frequencies[year]:
                if cluster in decade_frequencies[decade]:
                    decade_frequencies[decade][cluster] += cluster_frequencies[year][cluster]
                else:
                    decade_frequencies[decade][cluster] = cluster_frequencies[year][cluster]
        else:
            decade_frequencies[decade] = cluster_frequencies[year]

    # drop 2020
    decade_frequencies.pop(2020)

    # normalize the decade frequencies
    for decade in decade_frequencies:
        # get the total number of documents in the decade
        total_documents = 0
        for cluster in decade_frequencies[decade]:
            total_documents += decade_frequencies[decade][cluster]

        # normalize the cluster frequencies
        for cluster in decade_frequencies[decade]:
            decade_frequencies[decade][cluster] /= total_documents

    return decade_frequencies


# given a dictionary of the form {year: {cluster: relative frequency}},
# plot the relative frequencies of the clusters over time with years on
# the x-axis and relative frequencies on the y-axis

def plot_cluster_frequencies(cluster_frequencies: dict) -> None:
    """
    Given a dictionary of the form {year: {cluster: relative frequency}},
    plot the relative frequencies of the clusters over time with years on
    the x-axis and relative frequencies on the y-axis.

    Parameters
    ----------
    cluster_frequencies : A dictionary of the form {year: {cluster: relative frequency}}.

    Returns
    ----------
    None.
    """
    # create a dataframe from the dictionary
    df = pd.DataFrame.from_dict(cluster_frequencies, orient='index')

    # plot the dataframe
    df.plot(kind='bar', stacked=True, figsize=(20, 10), colormap='tab20')

    # set the title and axis labels
    plt.title('Relative Frequencies of Clusters Over Time')
    plt.xlabel('Year')
    plt.ylabel('Relative Frequency')

    # show the plot
    plt.show()