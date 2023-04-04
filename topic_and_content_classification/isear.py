"""
isear.py

Created on Mon Apr 3 2023

@author: Lukas

This file contains all methods for working with the ISEAR dataset.
"""

# import packages

import numpy as np
import pandas as pd

# functions for working with the ISEAR dataset

def load_isear() -> pd.core.frame.DataFrame:
    """
    Load the ISEAR dataset.

    Returns
    -------
    isear : The ISEAR dataset.
    """
    # mount the Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # load the dataset
    isear = pd.read_csv("/content/drive/MyDrive/Topic and Content Classification/DATA.csv")

    # keep only the columns 'SIT' and 'EMOT'
    isear = isear[['SIT', 'EMOT']]

    return isear


def get_isear_positives(dataset: pd.core.frame.DataFrame, emotion: int) -> list:
    """
    Get the indices of the positive examples of a given emotion from the ISEAR dataset.

    Parameters
    ----------
    dataset : The ISEAR dataset.

    emotion : The emotion to get the positive examples for.

    Returns
    -------
    positives : The indices of the positive examples of the given emotion.
    """
    # get the positive examples of the given emotion
    positives = dataset[dataset['EMOT'] == emotion].index.tolist()

    return positives


def load_isear_texts(isear: pd.core.frame.DataFrame, int_low: int, int_high: int) -> list:
    """
    Load the texts of the ISEAR dataset.

    Parameters
    ----------
    isear : The ISEAR dataset.

    int_low : The lower bound of the interval of texts to load.

    int_high : The upper bound of the interval of texts to load.

    Returns
    -------
    texts : The texts of the ISEAR dataset.
    """

    # get the texts of the ISEAR dataset
    texts = [isear['SIT'][i] for i in range(int_low, int_high)]

    return texts


def load_binary_isear_labels(isear: pd.core.frame.DataFrame, int_low: int, int_high: int, emot: int) -> list:
    """
    Load the labels of the ISEAR dataset with binary labels,
    i.e. 0 if the emotion is not the given emotion and 1 if the emotion is the given emotion.

    Parameters
    ----------
    isear : The ISEAR dataset.

    int_low : The lower bound of the interval of labels to load.

    int_high : The upper bound of the interval of labels to load.

    emot : The emotion to get the labels for.

    Returns
    -------
    labels : The labels of the ISEAR dataset.
    """
    # get the labels of the ISEAR dataset
    labels = [1 if isear['EMOT'][i] == emot else 0 for i in range(int_low, int_high)]
    
    return labels