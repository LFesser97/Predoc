"""
nltk_content_words.py

Created on Wed Jul 19 2023

@author: Lukas

This file contains all methods for mining content words using NLTK.
"""

# import packages

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Define functions

def remove_punctuation(documents: list) -> list:
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


def merge_across_linebreaks(documents: list) -> list:
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


def get_nltk_content_words(documents: list) -> list:
    """
    Given a list of documents, returns a list of content words.

    Parameters
    ----------
    documents : A list of documents, where each document is a string.

    Returns
    ----------
    A list of content words.
    """
    all_content_words = set()

    for document in documents:
        text = word_tokenize(document)
        pos_tags = nltk.pos_tag(text)

        # merge adjacent nouns, replace the two with the merged noun in the pos_tags list
        # repeat this process until no more merges are possible
        while True:
            merged = False
            for i in range(len(pos_tags) - 1):
                if pos_tags[i][1] in ['NN', 'NNS', 'NNP', 'NNPS'] and pos_tags[i+1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    pos_tags[i] = (pos_tags[i][0] + ' ' + pos_tags[i+1][0], 'NN')
                    pos_tags.pop(i+1)
                    merged = True
                    break
            if not merged:
                break

        # merge preceding adjectives with nouns, replace the two with the merged noun in the pos_tags list
        # repeat this process until no more merges are possible
        while True:
            merged = False
            for i in range(len(pos_tags) - 1):
                if pos_tags[i][1] in ['JJ', 'JJR', 'JJS'] and pos_tags[i+1][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    pos_tags[i] = (pos_tags[i][0] + ' ' + pos_tags[i+1][0], 'NN')
                    pos_tags.pop(i+1)
                    merged = True
                    break
            if not merged:
                break

        # get content words by selecting all nouns, i.e. 'NN', 'NNS', 'NNP', 'NNPS'
        content_words_raw = [word[0] for word in pos_tags if word[1] in ['NN', 'NNS', 'NNP', 'NNPS']]

        # remove stopwords
        content_words = clean_content_words(content_words_raw)

        # add to set of all content words
        all_content_words = all_content_words.union(content_words)

    return list(all_content_words)


def clean_content_words(content_words: set) -> set:
    """
    Given a set of content words, removes stopwords and words that are too short.

    Parameters
    ----------
    content_words : A set of content words.

    Returns
    ----------
    A set of content words that are not stopwords and are longer than 2 characters.
    """
    # convert content words to set
    content_words = set(content_words)
    
    # remove stopwords
    stop_words = set(stopwords.words('english'))

    # extend the stop words with the following words
    stop_words.union(set(['york', 'states', 'book', 'part', 'cambridge', 
                       'harvard', 'institution', 'press', 'even', 'princetion',
                       'united', 'known', 'chapter', 'chicago', 'brookings', 
                       'washington', 'oxford', 'paper', 'clarendon', 'hopkins',
                       'cent', 'wiley', 'chapters', 'publishing', 'would', 'first',
                       'american', 'review', 'bureau', 'per', 'em', 'great', 'years',
                       'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                        'nine', 'ten', 'zero', 'de', 'reviews', 'graduate', 'volume',
                        'year', 'author', 'authors', 'cent', 'way']))

    content_words = content_words.difference(stop_words)

    # remove words that are too short
    content_words = set([word for word in content_words if len(word) > 2])

    # MORE TO COME

    return content_words