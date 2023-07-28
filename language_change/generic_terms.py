"""
generic_terms.py

Created on Thu Jul 27 2023

@author: Lukas

This file contains all methods for filtering generic content words.
"""

# import packages

import numpy as np
import pandas as pd


def get_generic_terms(documents: list) -> dict:
    """
    Given a list of documents, identify the 20 most frequent content
    words in each document. Then take the set union over all documents
    and count, for each of the 20 words, the number of documents in
    which it appears. Then return a dictionary with the words as keys
    and the frequencies as values.

    Parameters
    ----------
    documents : A list of documents, where each document is a string.

    Returns
    ----------
    A dictionary with the most frequent content words as keys and
    the number of documents in which they appear as values.
    """
    document_content_words = {}

    for i in range(len(documents)):
        document = documents[i]
        baseline_content_words_set = set(get_nltk_content_words([document]))
        baseline_content_words_set = filter_numerical_values(filter_verbs(clean_content_words(list(baseline_content_words_set))))

        word_counts = {}

        # count the number of appearances of each content word in each document
        for word in baseline_content_words_set:
            word_counts[word] = word_counts.get(word, 0) + document.count(word)

        # sort the content words by their counts
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # get the 20 most frequent content words and add them to the dictionary
        document_content_words[i] = sorted_word_counts[:20]

    # get the set union of the 20 most frequent content words in each document
    content_words_set = set()
    for i in range(len(documents)):
        content_words_set = content_words_set.union(set(document_content_words[i]))

    # count the number of top 20 lists in which each content word appears
    content_words_dict = {}
    for word in content_words_set:
        for i in range(len(documents)):
            if word in document_content_words[i]:
                # if word is not in the dictionary, add it
                if word[0] not in content_words_dict:
                    content_words_dict[word[0]] = 1
                else:
                    content_words_dict[word[0]] += 1

    # sort the content words by their counts
    sorted_content_words_dict = sorted(content_words_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_content_words_dict


def aggregate_sorted_content_words(content_words: list) -> dict:
    """
    Given a list of tuples of the form (word, frequency), aggregate
    the frequencies of the words that are the same and return them in 
    order of decreasing frequency.

    Parameters
    ----------
    content_words : A list of tuples of the form (word, frequency).

    Returns
    ----------
    A dictionary with the words as keys and the frequencies as values.
    """
    content_words_dict = {}
    for word in content_words:
        if word[0] not in content_words_dict:
            content_words_dict[word[0]] = word[1]
        else:
            content_words_dict[word[0]] += word[1]

    # sort the content words by their counts
    sorted_content_words_dict = sorted(content_words_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_content_words_dict