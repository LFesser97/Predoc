"""
load_datasets.py

Created on Tue Mar 21 2023

@author: Lukas

This file contains all functions for loading Japanese datasets.
"""

# import packages

import pandas as pd
import numpy as np
import json


# functions for loading in datasets and storing them in a pandas dataframe

def load_from_json(filename, dataframe):
    """
    This function loads a Japanese dataset from a .json file.

    Parameters
    ----------
    filename : str
        The filename of the Japanese dataset.

    The json is in the format: {
    "title": "",
    "compnay_type": "",
    "address": "",
    "variables": [{"key": "", "tag": "est_date", "values": [""]},
                  {"key": "", "tag": "n_stocks_authorized", "values": [""]},
                  {"key": "", "tag": "stock_capital", "values": [""]},
                  {"key": "", "tag": "accounting_period", "values": [""]},
                  {"key": "", "tag": "dividend", "values": ["", "", ""]},
                  {"key": "", "tag": "personnel", "values": ["", "", ...]},
                  {"key": "", "tag": "main_shareholders", "values": ["", "", ...]},
                  {"key": "", "tag": "n_employee", "values": [""]}, 
                  {"key": "", "tag": "banks", "values": [""]},
                  {"key": "", "tag": "locations", "values": ["", "", ...]},
                ]
    }

    dataframe : pandas dataframe
        The dataframe to which the company should be appended.

    Returns
    -------
    dataset : pandas dataframe
        The dataset as a pandas dataframe.

    The dataframe has the following columns:
        - title
        - company_type
        - address
        - est_date
        - n_stocks_authorized
        - stock_capital
        - accounting_period
        - dividend
        - personnel
        - main_shareholders
        - n_employee
        - banks
        - locations
    """
    # load json file using json.load()
    with open(filename) as f:
        dataset = json.load(f)

    # only have one company per file
    company_information = {"title": dataset["title"],
                           "company_type": dataset["compnay_type"],
                           "address": dataset["address"]}

    # get list of available variables
    available_variables = [dataset["variables"][i]["tag"] for i in range(len(dataset["variables"]))]

    # get values for each variable
    for variable in available_variables:
        company_information[variable] = dataset["variables"][available_variables.index(variable)]["values"]

    # insert NaN for variables that are not available, but are in the dataframe
    for column in dataframe.columns:
        if column not in available_variables:
            company_information[column] = np.nan

    # concatenate the company information to the dataframe using pd.concat()
    dataframe = pd.concat([dataframe, pd.DataFrame_from_dict(company_information, index=[0])], ignore_index=True)

    return df