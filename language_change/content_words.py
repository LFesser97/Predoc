"""
content_words.py

Created on Thu Jul 13 2023

@author: Lukas

This file contains two methods for identifying content words.
"""

# import packages

import numpy as np
import pandas as pd
import nltk

# functions for identifying content words

def find_common_unigrams(text: str, n: int) -> list:
    """
    Given a text, find the n most common unigrams.

    Parameters
    ----------
    text : The text to find the most common unigrams for.

    n : The number of most common unigrams to find.

    Returns
    -------
    common_unigrams : The n most common unigrams in the text.
    """
    # create a dictionary for the unigrams
    unigrams = {}

    # for each unigram in the text
    for unigram in text.split():
        # if the unigram is not in the dictionary, add it
        if unigram not in unigrams:
            unigrams[unigram] = 1
        # if the unigram is in the dictionary, increase its count by 1
        else:
            unigrams[unigram] += 1

    # sort the unigrams by their count
    sorted_unigrams = sorted(unigrams.items(), key=lambda x: x[1], reverse=True)

    # return the n most common unigrams
    return [unigram[0] for unigram in sorted_unigrams[:n]]



def find_common_bigrams(text: str, n: int) -> list:
    """
    Given a text, find the n most common bigrams.

    Parameters
    ----------
    text : The text to find the most common bigrams for.

    n : The number of most common bigrams to find.

    Returns
    -------
    common_bigrams : The n most common bigrams in the text.
    """
    # create a dictionary for the bigrams
    bigrams = {}

    # for each bigram in the text
    for bigram in nltk.bigrams(text.split()):
        # if the bigram is not in the dictionary, add it
        if bigram not in bigrams:
            bigrams[bigram] = 1
        # if the bigram is in the dictionary, increase its count by 1
        else:
            bigrams[bigram] += 1

    # sort the bigrams by their count
    sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)

    # return the n most common bigrams
    return [bigram[0] for bigram in sorted_bigrams[:n]]


def find_common_bigrams_in_corpus(corpus: list, n: int) -> list:
    """
    Given a corpus, find the n most common bigrams.

    Parameters
    ----------
    corpus : The corpus to find the most common bigrams for.

    n : The number of most common bigrams to find.

    Returns
    -------
    common_bigrams : The n most common bigrams in the corpus.
    """
    # create a dictionary for the bigrams
    bigrams = {}

    # for each document in the corpus
    for document in corpus:
        # for each bigram in the document
        for bigram in nltk.bigrams(document.split()):
            # if the bigram is not in the dictionary, add it
            if bigram not in bigrams:
                bigrams[bigram] = 1
            # if the bigram is in the dictionary, increase its count by 1
            else:
                bigrams[bigram] += 1

    # sort the bigrams by their count
    sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)

    # return the n most common bigrams
    return [bigram[0] for bigram in sorted_bigrams[:n]]


def filter_unigrams(unigrams: list, stop_words: list) -> list:
    """
    Given a list of unigrams, filter out the stop words.

    Parameters
    ----------
    unigrams : The list of unigrams to filter.

    stop_words : The list of stop words to filter out.

    Returns
    -------
    filtered_unigrams : The filtered list of unigrams.
    """
    # return the filtered list of unigrams
    return [unigram for unigram in unigrams if unigram not in stop_words]


def get_nlkt_stop_words() -> list:
    """
    Get the stop words from the nltk package.

    Returns
    -------
    stop_words : The list of stop words.
    """
    # import the stop words from the nltk package
    from nltk.corpus import stopwords

    # return the stop words
    return stopwords.words('english')


def docs_to_lowercase(documents: list) -> list:
    """
    Convert a list of documents to lowercase.

    Parameters
    ----------
    documents : The list of documents to convert to lowercase.

    Returns
    -------
    documents_lowercase : The list of documents in lowercase.
    """
    # return the list of documents in lowercase
    return [document.lower() for document in documents]


def remove_line_breaks(documents: list) -> list:
    """
    Remove line breaks from a list of documents,
    where linebreaks are of the form '- '.

    Parameters
    ----------
    documents : The list of documents to remove line breaks from.

    Returns
    -------
    documents_without_line_breaks : The list of documents without line breaks.
    """
    # return the list of documents without line breaks
    return [document.replace('- ', '') for document in documents]


def remove_abbreviations(words: list) -> list:
    """
    Filter out abbreviations from a list of words,
    where abbreviations are of the form 'word.'.

    Parameters
    ----------
    words : The list of words to filter abbreviations from.

    Returns
    -------
    words_without_abbreviations : The list of words without abbreviations.
    """
    # return the list of words without abbreviations
    return [word for word in words if not word.endswith('.')]


def remove_numerical_words(words: list) -> list:
    """
    Filter out numerical words from a list of words,
    where numerical words are one of the following:
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'.

    Parameters
    ----------
    words : The list of words to filter numerical words from.

    Returns
    -------
    words_without_numerical_words : The list of words without numerical words.
    """
    # return the list of words without numerical words
    return [word for word in words if word not in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']]


def bold_words(text, words):
    """
    Given a text and a list of words, bold the words in the text,
    but only if they match exactly. For example, given the text
    'This is socialism', and the words ['is', 'social'], the 
    function will return 'This **is** socialism', and not
    'This **is** **social**ism'.

    Parameters
    ----------
    text : The text to bold the words in.

    words : The list of words to bold in the text.

    Returns
    -------
    text_bolded : The text with the words bolded.
    """
    # split the text into words
    text_split = text.split()

    # for each word in the text
    for i, word in enumerate(text_split):
        # if the word is in the list of words
        if word in words:
            # bold the word
            text_split[i] = '**' + word + '**'

    # return the text with the words bolded
    return ' '.join(text_split)


def bold_bigrams(text, bigrams):
    """
    Given a text and a list of bigrams, bold the bigrams in the text,
    but only if they match exactly. For example, given the text
    'This is socialism', and the bigrams [('is', 'social'), ('social', 'ism')],
    the function will return 'This **is socialism**', and not
    'This **is** **social**ism'.

    Parameters
    ----------
    text : The text to bold the bigrams in.

    bigrams : The list of bigrams to bold in the text.

    Returns
    -------
    text_bolded : The text with the bigrams bolded.
    """
    # split the text into words
    text_split = text.split()

    # for each bigram in the text
    for i, bigram in enumerate(nltk.bigrams(text_split)):
        # if the bigram is in the list of bigrams
        if bigram in bigrams:
            # bold the bigram
            text_split[i] = '**' + bigram[0] + ' ' + bigram[1] + '**'

    # return the text with the bigrams bolded
    return ' '.join(text_split)


def remove_stopwords_from_bigrams(bigrams: list, stopwords: list) -> list:
    """
    Given a list of bigrams and a list of stopwords,
    remove the stopwords from the bigrams.

    Parameters
    ----------
    bigrams : The list of bigrams to remove stopwords from.

    stopwords : The list of stopwords to remove from the bigrams.

    Returns
    -------
    bigrams_without_stopwords : The list of bigrams without stopwords.
    """
    # return the list of bigrams without stopwords
    return [bigram for bigram in bigrams if bigram[0] not in stopwords and bigram[1] not in stopwords]


def remove_abbreviations_from_bigrams(bigrams: list) -> list:
    """
    Given a list of bigrams, remove the abbreviations from the bigrams.

    Parameters
    ----------
    bigrams : The list of bigrams to remove abbreviations from.

    Returns
    -------
    bigrams_without_abbreviations : The list of bigrams without abbreviations.
    """
    # return the list of bigrams without abbreviations
    return [bigram for bigram in bigrams if not bigram[0].endswith('.') and not bigram[1].endswith('.')]


def remove_universities_from_bigrams(bigrams: list) -> list:
    """
    Given a list of bigrams, remove the universities from the bigrams.

    Parameters
    ----------
    bigrams : The list of bigrams to remove universities from.

    Returns
    -------
    bigrams_without_universities : The list of bigrams without universities.
    """
    # return the list of bigrams without universities
    return [bigram for bigram in bigrams if bigram[0] not in ['university', 'universities'] and bigram[1] not in ['university', 'universities']]