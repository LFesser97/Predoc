"""
filtering.py

Created on Tue Aug 29 2023

@Author: Lukas

This file contains all methods for filtering content words.
"""

# import packages
import numpy as np
import pandas as pd
import json

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from glob import glob

import pickle
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# methods for filtering content words

class filtering:

    @staticmethod
    def _clean_content_words(content_words: list) -> set:
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
        content_words = set([word for word in content_words if len(word) > 5])

        return content_words
    

    @staticmethod
    def _filter_verbs(content_words: set) -> set:
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


    @staticmethod
    def _filter_numerical_values(content_words: set) -> set:
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