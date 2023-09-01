"""
get_content_words.py

Created on Tue Aug 29 2023

@Author: Lukas

This file contains all methods for mining content words.
"""

# import packages
import json
import numpy as np
import pandas as pd
import argparse
import multiprocessing
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# import methods from other files
from preprocessing import preprocessing, jstor_preprocessing
from filtering import filtering


# main functions for mining content words

class content_words:

    @staticmethod
    def _read_input_file(input_file: str, jstor: bool) -> list:
        """
        Given a path to an input file, reads the file and returns a list of documents.

        Parameters
        ----------
        input_file : The path to the input file.
        jstor : Indicates whether the input file is a jstor file.

        Returns
        ----------
        A list of documents, where each document is a string.
        """
        # read the input file
        if jstor:
            documents = preprocessing._process_data_json(input_file)
        else:
            # load the data using json
            with open(input_file, 'r') as f:
                documents = json.load(f)

            assert isinstance(documents, list), "The input file must be a list of documents."

        return documents
    

    @staticmethod
    def __get_aggregate_content_words(documents: list) -> list:
        """
        Given a list of documents, identify content words.
        Then return a dictionary with the documents ids as keys
        and a list of content words as values.
        """
        spell = SpellChecker()

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

            # add to set of all content words
            all_content_words = all_content_words.union(content_words)

        return list(all_content_words)
    

    @staticmethod
    def get_content_words(documents: list, aggregate: bool) -> dict:
        """
        Given a list of documents, identify content words.
        Then return a dictionary with the documents ids as keys.
        """
        if aggregate:
            content_words = {0 : content_words.__get_aggregate_content_words(documents)}
        else:
            content_words = {i : content_words.__get_aggregate_content_words([documents[i]]) for i in range(len(documents))}

        return content_words
    

    @staticmethod
    def _filter_documents(content_words: list) -> list:
        """
        Given a list of content words,
        - ...
        """
        stopwords_removed = filtering._clean_content_words(content_words)
        verbs_removed = filtering._filter_verbs(stopwords_removed)
        numerical_values_removed = filtering._filter_numerical_values(verbs_removed)

        return list(numerical_values_removed)


# if the file is run as a script, run the following code
if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Mine content words from a list of documents.')
    parser.add_argument('input_file', type=str, help='The path to the input file.')
    parser.add_argument('output_file', type=str, help='The path to the output file.')
    parser.add_argument('--jstor', dest='jstor', action='store_true', help='Indicates whether the input file is a jstor file.')
    parser.add_argument('--document', dest='document', action='store_true', help='Indicates whether content words should be collected for each document.')

    args = parser.parse_args()

    # read the input file
    documents = content_words._read_input_file(args.input_file, args.jstor)

    # get content words
    # use multi-processing to speed up the process, i.e. use all available cores
    # to process the documents in parallel
    pool = multiprocessing.Pool()
    content_words_raw = pool.map(content_words.get_content_words, documents, not args.document)
    pool.close()
    pool.join()
    
    content_words_filtered = content_words._filter_documents(content_words_raw)

    # write the output file
    with open(args.output_file, 'w') as f:
        json.dump(content_words_filtered, f)