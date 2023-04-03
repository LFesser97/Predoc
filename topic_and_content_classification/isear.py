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