"""
load_datasets.py

Created on Tue Mar 21 2023

@author: Lukas

This file contains all functions for loading Japanese datasets.
"""

# import packages

import pandas as pd
import numpy as np


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
        The dataframe to which the dataset should be appended.

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
    # load json file
    dataset = pd.read_json(filename)

    # iterate over all companies
    for i in range(len(dataset)):
        # get company
        company = dataset.iloc[i]

        # get title
        title = company["title"]

        # get company_type
        company_type = company["company_type"]

        # get address
        address = company["address"]

        # get est_date
        est_date = company["variables"][0]["values"][0]

        # get n_stocks_authorized
        n_stocks_authorized = company["variables"][1]["values"][0]

        # get stock_capital
        stock_capital = company["variables"][2]["values"][0]

        # get accounting_period
        accounting_period = company["variables"][3]["values"][0]

        # get dividend
        dividend = company["variables"][4]["values"]

        # get personnel
        personnel = company["variables"][5]["values"]

        # get main_shareholders
        main_shareholders = company["variables"][6]["values"]

        # get n_employee
        n_employee = company["variables"][7]["values"][0]

        # get banks
        banks = company["variables"][8]["values"][0]

        # get locations
        locations = company["variables"][9]["values"]

        # get accounting_period
        accounting_period = company["variables"][10]["values"]

        # append to dataframe
        df = dataframe.append({"title": title, "company_type": company_type, "address": address, "est_date": est_date, "n_stocks_authorized": n_stocks_authorized, "stock_capital": stock_capital, "accounting_period": accounting_period, "dividend": dividend, "personnel": personnel, "main_shareholders": main_shareholders, "n_employee": n_employee, "banks": banks, "locations": locations, "accounting_period": accounting_period}, ignore_index=True)

    return df