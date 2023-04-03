"""
keyword_retrieval.py

Created on Mon Apr 3 2023

@author: Lukas

This file contains all methods for keyword-based text retrieval.
"""

# import packages

import numpy as np
import os

# functions for keyword-based text retrieval

def keyword_retrieval(keywords: list, corpus_folder: str) -> list:
    """
    A function that retrieves documents based on keywords.

    Parameters
    ----------
    keywords : The keywords to retrieve documents for.

    corpus_folder : The folder containing the corpus.

    Returns
    -------
    documents : The documents retrieved based on the keywords.
    """
    # create a list of all files in the corpus folder
    files = os.listdir(corpus_folder)

    # for each file, check whether it contains at least one of the keywords
    documents = []
    for file in files:
        with open(corpus_folder + file, "r") as f:
            text = f.read()
            for keyword in keywords:
                if keyword in text:
                    documents.append(file)
                    break

    return documents