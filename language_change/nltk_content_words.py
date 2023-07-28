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
from spellchecker import SpellChecker

spell = SpellChecker()


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

        unknown_words = []

        # for each noun, check if it is contained in the spellchecker dictionary
        # by using spell.unknown(). If not, add it to the list of unknown words.
        for word, tag in pos_tags:
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                if word.lower() in spell.unknown([word.lower()]):
                    # merge it with the next word and check if the merged word is in the spellchecker dictionary
                    if pos_tags.index((word, tag)) < len(pos_tags) - 1:
                        next_word = pos_tags[pos_tags.index((word, tag)) + 1][0]
                        merged_word = word + '' + next_word
                        if merged_word.lower() in spell.unknown([merged_word.lower()]):
                            unknown_words.append(word)

        # get content words by selecting all nouns, i.e. 'NN', 'NNS', 'NNP', 'NNPS'
        content_words_raw = [word[0] for word in pos_tags if word[1] in ['NN', 'NNS', 'NNP', 'NNPS']]

        # remove stopwords
        content_words = clean_content_words(content_words_raw)

        # remove words that are in content_words_raw but not in content_words from pos_tags
        pos_tags = [word for word in pos_tags if not (word[0] in content_words_raw and word[0] not in content_words)]  


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

        # clean content words
        content_words = clean_content_words([word[0] for word in pos_tags if word[1] in ['NN', 'NNS', 'NNP', 'NNPS']])

        # add to set of all content words
        all_content_words = all_content_words.union(content_words)

    return list(all_content_words)


def clean_content_words(content_words: list) -> set:
    """
    Given a set of content words, removes stopwords and words that are too short.

    Parameters
    ----------
    content_words : A list of content words.

    Returns
    ----------
    A set of content words that are not stopwords and are longer than 2 characters.
    """
    # convert content words to set
    content_words = set(content_words)

    # convert all words to lowercase
    content_words = set([word.lower() for word in content_words])

    # remove stopwords
    stop_words = set(stopwords.words('english'))

    # extend the stop words with the following words
    other_words = set(['york', 'states', 'book', 'part', 'cambridge', 
                       'harvard', 'institution', 'press', 'even', 'princeton',
                       'united', 'known', 'chapter', 'chicago', 'brookings', 
                       'washington', 'oxford', 'paper', 'clarendon', 'hopkins',
                       'cent', 'wiley', 'chapters', 'publishing', 'would', 'first',
                       'american', 'review', 'bureau', 'per', 'em', 'great', 'years',
                       'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                        'nine', 'ten', 'zero', 'de', 'reviews', 'graduate', 'volume',
                        'year', 'author', 'authors', 'cent', 'way', 'analysis',
                        'period', 'problem', 'economy', 'economic', 'percent', 'equation',
                        'example', 'countries', 'journal', 'university', 'parameter',
                        'problems', 'discussion', 'solution', 'relation', 'country',
                        'difference', 'coefficient', 'regression', 'method', 'economists',
                        'equations', 'matrix', 'outcome', 'literature', 'nation', 'capita',
                        'assumptions', 'standard', 'others', 'america', 'europe', 
                        'parameters', 'argument', 'conclusion', 'economies', 'amsfonts',
                        'stmaryrd', 'xspace', 'fontenc', 'amsmath', 'amssymb', 'aastex',
                        'mathrsfs', 'wncyss', 'amsbsy', 'portland', 'region', 'sumption',
                        'person', 'concept', 'alternative', 'coefficients', 'london',
                        'statistics', 'librium', 'correlation', 'periods', 'outcomes',
                        'success', 'equilib', 'addition', 'economist', 'methods'])


    content_words = set([word for word in content_words if word not in stop_words])

    content_words = set([word for word in content_words if word not in other_words])

    content_words = set([word for word in content_words if nltk.pos_tag([word])[0][1] in ['NN', 'NNS', 'NNP', 'NNPS']])

    # remove words that are too short
    content_words = set([word for word in content_words if len(word) > 5])

    # remove verbs
    content_words = filter_verbs(content_words)

    return content_words


def filter_verbs(content_words: set) -> set:
    """
    Given a set of content words, removes verbs,
    by using wordnet to check if the words can be lemmatized to a verb.

    Parameters
    ----------
    content_words : A set of content words.

    Returns
    ----------
    A set of content words that are not verbs.
    """
    # convert content words to a list
    content_words = list(content_words)

    for word in content_words:
        synsets = wordnet.synsets(word)

        for synset in synsets:
            if synset.pos() == 'v':
                content_words.remove(word)
                break

    return set(content_words)


def filter_numerical_values(content_words: set) -> set:
    """
    Filter out content words that contain a numerical value,
    such as '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'.

    Parameters
    ----------
    content_words : A set of content words.

    Returns
    ----------
    A set of content words that do not contain a numerical value.
    """
    content_words = list(content_words)

    for word in content_words:
        if any(char.isdigit() for char in word):
            content_words.remove(word)

    return set(content_words)


def combine_text_parts(text: list) -> str:
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


def preprocess_docs(documents: list) -> list:
    """
    Given a list of documents, preprocesses them by combining the text parts
    in each document, changing every document to lowercase, and removing line breaks.

    Parameters
    ----------
    documents : A list of documents.

    Returns
    ----------
    A list of preprocessed documents.
    """
    preprocessed_docs = []
    for document in documents:
        # combine text parts
        document = combine_text_parts(document)

        # change to lowercase
        document = document.lower()

        # remove line breaks
        document = document.replace('\n', ' ')
        document = document.replace('- ', '')

        preprocessed_docs.append(document)

    return preprocessed_docs