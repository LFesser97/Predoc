
"""
preprocessing.py

Created on Tue Aug 29 2023

@Author: Lukas

This file contains all methods for preprocessing the data.
"""

# import packages
import json
import numpy as np
import pandas as pd


# preprocessing a list of documents

class preprocessing:

    @staticmethod
    def _remove_punctuation(documents: list) -> list:
        """
        Given a list of documents, removes all punctuation from each document.
        Punctuation includes all symbols in the string "symbols" below.

        Parameters
        ----------
        documents : A list of documents, where each document is a string.

        Returns
        ----------
        A list of documents, where each document is a string with no punctuation.
        """
        symbols = "!\"#$%&()*+-,/:;<=>?@[\]'^_`{|}~"

        no_punct_documents = []
        for document in documents:
            for i in symbols:
                document = document.replace(i, '')
            if len(document) > 0:
                no_punct_documents.append(document)

        return no_punct_documents
    

    @staticmethod
    def remove_references(documents: list) -> list:
        """
        Given a list of documents, removes the references section from each document.
        """
        # set every document to lowercase
        documents = [document.lower() for document in documents]

        # find the last occurrence of the word "references" or "bibliography" in each document
        references_indices = []
        for document in documents:
            references_indices.append(max(document.rfind('references'), document.rfind('bibliography')))

        # remove the references section from each document
        no_references_documents = []
        for i in range(len(documents)):
            no_references_documents.append(documents[i][:references_indices[i]])

        return no_references_documents
    

    @staticmethod
    def prepare_full_jstor_data(jstor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe with JSTOR data, keep only the columns 'id',
        'title', 'abstract', 'fulltext', 'publicationYear', and 'creator'.
        Rename the columns 'publicationYear' and 'creator' to 'year' and 'author'.
        """
        jstor_df = jstor_df[['id', 'title', 'abstract', 'fullText', 'publicationYear', 'creator']]
        jstor_df = jstor_df.rename(columns={'publicationYear': 'year', 'creator': 'author'})

        return jstor_df


    @staticmethod
    def _merge_across_linebreaks(documents: list) -> list:
        """
        Given a list of documents, merges two parts of a word
        that are separated by a linebreak.

        Parameters
        ----------
        documents : A list of documents, where each document is a string.

        Returns
        ----------
        A list of documents, where each document is a string with no linebreaks.
        """
        merged_documents = []
        for document in documents:
            merged_documents.append(document.replace('-\n', ''))

        return merged_documents
    

# preprocessing jstor documents

class jstor_preprocessing:

    @staticmethod
    def _process_data_json(data_json_path: str) -> pd.DataFrame:
        """
        Given a path to a json file with JSTOR data, return a pandas dataframe
        """
        with open(data_json_path) as f:
            data = f.readlines()

        data_json = [json.loads(item) for item in data]

        processed_data = []
        for item in data_json:
            if "creator" in item:
                processed_data.append({
                    "id": item["id"],
                    "title": item["title"],
                    "year": item["publicationYear"],
                    "abstract": item["abstract"],
                    "author": item["creator"],
                    "fulltext": item["fullText"]
                })
            else:
                processed_data.append({
                    "id": item["id"],
                    "title": item["title"],
                    "year": item["publicationYear"],
                    "abstract": item["abstract"],
                    "author": None,
                    "fulltext": item["fullText"]
                })

        jstor_df = pd.DataFrame.from_dict(processed_data)

        return jstor_df
    

    @staticmethod
    def _get_documents_from_path(data_json_path: str) -> list:
        """
        Given a path to a json file with JSTOR data, return a list of documents.
        """
        jstor_df = preprocessing._process_data_json(data_json_path)
        documents = []
        for fulltext in jstor_df['fulltext']:
            documents.append(fulltext)

        return documents


    @staticmethod
    def _combine_text_parts(text: list) -> str:
        """
        Given a list of text parts, combines them into a single string.

        Parameters
        ----------
        text : A list of text parts.

        Returns
        ----------
        A string containing all text parts.
        """
        combined_text = ''
        for part in text:
            combined_text += part + ' '

        return combined_text