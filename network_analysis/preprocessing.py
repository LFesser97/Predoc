"""
preprocessing.py

Created on Wed Mar 22 2023

@author: Lukas

This file contains all functions for preprocessing Japanese datasets.
"""

# import packages

import pandas as pd
import numpy as np
import json


# functions for preprocessing datasets

def load_supplier_from_csv(filename, dataframe):
    """
    This function loads in a csv file, extracts the supplier variables
    and appends them to the dataframe.

    Parameters
    ----------
    filename : str
        The filename of the Japanese dataset to be loaded.

    dataframe : pandas dataframe
        The dataframe to which the information should be appended.

    Returns
    -------
    df_list : pandas dataframe
        The updated dataset as a pandas dataframe.
    """
    # get the source -> target pairs
    df = pd.read_csv(filename)
    df = df[['partner_main_title', 'partner_type', 'matched_tk_text']]
    df.replace(-9, np.nan, inplace=True)
    df_list = df.dropna()

    df_list['supplier'] = df.apply(
        lambda x: x['partner_main_title'] if x['partner_type'] == 'fanmai' else x['matched_tk_text'],
        axis=1)

    df_list['title'] = df.apply(
        lambda x: x['matched_tk_text'] if x['partner_type'] == 'fanmai' else x['partner_main_title'],
        axis=1)

    df_list = df_list[['supplier', 'title']]

    # add all other columns in dataframe not in df_list to df_list
    for column in dataframe.columns:
        if column not in df_list.columns:
            df_list[column] = np.nan

    return df_list