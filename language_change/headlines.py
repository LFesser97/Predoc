"""
headlines.py

Created on Tue Aug 22 2023

@author: Lukas

This file contains all the 
"""

# import packages

import nltk
nltk.download('words')
from nltk.corpus import words

english_words = set(words.words())


# auxiliary functions

class headlines:
    """
    A class with static methods for pulling content words from headlines.
    """
    @staticmethod
    def filter_non_dictionary_words(input_words: list) -> list:
        """
        Given a list of words, return a list of words that are in the
        English dictionary.
        """
        return [word for word in input_words if word in english_words]
    

    @staticmethod
    def get_headline_indices(target_words: list, headlines: dict) -> dict:
        """
        Given a list of target words and a dictionary of headlines,
        return a dictionary with the target words as keys and a dictionary
        with years as keys and indices of headlines containing the target
        word as values.
        """
        headline_indices = {}

        for word in target_words:
            word_indices = {}
            for decade in headlines.keys():
                word_indices[decade] = [i for i in range(len(headlines[decade])) if word in headlines[decade][i]]
            headline_indices[word] = word_indices

        return headline_indices
    

    @staticmethod
    def get_headlines_from_indices(indices: dict, embeddings: dict, headlines: dict) -> dict:
        """
        Given alist with target words and a dictionary
        with years as keys and indices of headlines containing the target
        word as values, return a dictionary with the target words as keys
        and a dictionary with years as keys and headlines containing the
        target word as values.
        """
        target_headlines = {}

        for word in indices.keys():
            word_headlines = {}
            for decade in indices[word].keys():
                word_headlines[decade] = [[headlines[decade][i], embeddings[decade][i], i] for i in indices[word][decade]]
            target_headlines[word] = word_headlines

        return target_headlines


    @staticmethod
    def get_nearest_neighbors(target_headlines: dict, definitions: dict) -> dict:
        """
        Given a dictionary with embeddings under target_headlines[word][decade][i][1]
        and a dictionary with a list of embeddings for definitions under
        definitions[word], return an updated target_headlines dictionary
        with the nearest neighbors of the target words under
        target_headlines[word][decade][i][3].
        """
        for word in target_headlines.keys():
            for decade in target_headlines[word].keys():
                for i in range(len(target_headlines[word][decade])):
                    target_headlines[word][decade][i].append(headlines.get_nearest_neighbor(target_headlines[word][decade][i][1], definitions[word]))

        return target_headlines


    @staticmethod
    def get_nearest_neighbor(embedding: list, definitions: list) -> list:
        """
        Given a list of embeddings and a list of definitions, return the
        definition with the nearest embedding.
        """
        distances = []
        for definition in definitions:
            distances.append(headlines.get_distance(embedding, definition))
        return definitions[distances.index(min(distances))]
    

    @staticmethod
    def get_cosine_distance(embedding1: list, embedding2: list) -> float:
        """
        Given two embeddings, return the cosine distance between them.
        """
        return 1 - headlines.get_cosine_similarity(embedding1, embedding2)
    

    @staticmethod
    def get_cosine_similarity(embedding1: list, embedding2: list) -> float:
        """
        Given two embeddings, return the cosine similarity between them.
        """
        return sum([embedding1[i] * embedding2[i] for i in range(len(embedding1))]) / (headlines.get_norm(embedding1) * headlines.get_norm(embedding2))
    

    @staticmethod
    def get_norm(embedding: list) -> float:
        """
        Given an embedding, return its norm.
        """
        return sum([embedding[i] ** 2 for i in range(len(embedding))]) ** 0.5
    

    @staticmethod
    def create_usage_histogram(target_headlines: dict) -> None:
        """
        Given a dictionary with cluster id under
        nn_dict[word][decade][i][3] for document i, create a usage
        histogram for each word.
        """
        for word in target_headlines.keys():
            for decade in target_headlines[word].keys():
                cluster_ids = [target_headlines[word][decade][i][3] for i in range(len(target_headlines[word][decade]))]

                # create a dictionary with cluster ids as keys and counts as values
                cluster_id_counts = {}
                for cluster_id in cluster_ids:
                    if cluster_id not in cluster_id_counts:
                        cluster_id_counts[cluster_id] = 1
                    else:
                        cluster_id_counts[cluster_id] += 1

        return cluster_id_counts