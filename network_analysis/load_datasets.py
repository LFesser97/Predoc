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
                  {"key": "", "tag": "accounting_period", "values": ["", "", ...]},
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
        - accounting_period
    """
    # load json file using json.load()
    with open(filename) as f:
        dataset = json.load(f)

    # only have one company per file
    title = dataset["title"]
    company_type = dataset["compnay_type"]
    address = dataset["address"]
    est_date = dataset["variables"][0]["values"][0]
    n_stocks_authorized = dataset["variables"][1]["values"][0]
    stock_capital = dataset["variables"][2]["values"][0]
    accounting_period = dataset["variables"][3]["values"][0]
    dividend = dataset["variables"][4]["values"]
    personnel = dataset["variables"][5]["values"]
    main_shareholders = dataset["variables"][6]["values"]
    n_employee = dataset["variables"][7]["values"][0]
    banks = dataset["variables"][8]["values"][0]
    locations = dataset["variables"][9]["values"]
    accounting_period = dataset["variables"][10]["values"]

    # append to dataframe
    df = dataframe.append({"title": title, "dataset_type": company_type,
                           "address": address, "est_date": est_date,
                           "n_stocks_authorized": n_stocks_authorized,
                           "stock_capital": stock_capital, "accounting_period": accounting_period,
                           "dividend": dividend, "personnel": personnel,
                           "main_shareholders": main_shareholders, "n_employee": n_employee,
                           "banks": banks, "locations": locations,
                           "accounting_period": accounting_period},
                           ignore_index=True)

    return df