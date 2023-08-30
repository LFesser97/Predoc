"""
embeddings.py

Created on Wed Aug 30 2023

@Author: Lukas

This file contains all methods for creating embeddings.
"""

# import packages
import json
import numpy as np
import pandas as pd
import argparse
import pickle
import torch

from tqdm import tqdm


class embeddings
    @staticmethod
    def get_bert_embeddings(documents: list) -> list:
        """
        Given a list of documents, returns a list of embeddings for each document.
        The embeddings are created using the bert-as-service library.
        """
        # import packages
        from bert_serving.client import BertClient

        # start the bert server
        bc = BertClient()

        # create the embeddings
        embeddings = bc.encode(documents)

        return embeddings

    
    @staticmethod
    def get_sbert_embeddings(documents: list) -> list:
        """
        Given a list of documents, returns a list of embeddings for each document.
        The embeddings are created using the sentence-transformers library.
        """
        # import packages
        from sentence_transformers import SentenceTransformer

        # load the model
        model = SentenceTransformer('bert-base-nli-mean-tokens')

        # create the embeddings
        embeddings = model.encode(documents)

        return embeddings
    

if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description='Create embeddings.')
    parser.add_argument('input_file', type=str, help='The path to the input file.')
    parser.add_argument('output_file', type=str, help='The path to the output file.')
    parser.add_argument('embedding_type', type=str, help='The type of embeddings to create.')
    args = parser.parse_args()

    # read the input file
    with open(args.input_file, 'rb') as f:
        documents = pickle.load(f)

    # create the embeddings, assert that the input file is a list of strings
    assert isinstance(documents, list), "The input file must be a list."
    assert isinstance(documents[0], str), "The input file must be a list of strings."

    if args.embedding_type == 'bert':
        embeddings = embeddings.get_bert_embeddings(documents)
    elif args.embedding_type == 'sbert':
        embeddings = embeddings.get_sbert_embeddings(documents)

    # write the output file
    with open(args.output_file, 'wb') as f:
        pickle.dump(embeddings, f)