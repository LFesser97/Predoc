"""
linguistic_drifts.py

Created on Thu Jul 6 2023

@author: Lukas

This file contains all methods for computing measures of linguistic drifts.
"""


# import packages

import numpy as np
import pandas as pd


class CorporaComparison:
    def __init__(self, corpus1: pd.DataFrame, corpus2: pd.DataFrame):
        """
        This class compares two corpora and computes various linguistic drifts between them.

        NOTE: We assume that the corpora are pandas dataframes with the following columns:
        - 'name/id'
        - 'text' (the text of the corpus as a string)
        - 'tokens' (the text of the corpus as a list of tokens)
        - 'embedding' (the SBERT embedding of the corpus as a list of floats)
        """
        self.corpus1 = corpus1
        self.corpus2 = corpus2
        self.content_words = []
        self.content_word_counts = {}
        self.pos_5gram_model = None
        self.sbert_model = None


    def add_content_words(self, words: list):
        """
        Adds a list of content words to the CorporaComparison object.

        Parameters
        ----------
        words : A list of content words.
        """
        self.content_words.append(words)


    def count_content_words(self):
        """
        Counts the number of content words in each corpus and 
        adds the results to the CorporaComparison object.
        """
        for word in self.content_words:
            self.content_word_counts[word] = [0, 0]
            for i in range(len(self.corpus1)):
                self.content_word_counts[word][0] += self.corpus1['text'][i].count(word)
            for i in range(len(self.corpus2)):
                self.content_word_counts[word][1] += self.corpus2['text'][i].count(word)

    
    def return_voc_drift(self) -> float:
        """
        Return the average vocabulary drift between corpus 1 and the texts in corpus 2.

        Returns
        -------
        voc_drift : The average vocabulary drift between corpus 1 and the texts in corpus 2.
        """
        if len(self.content_word_counts) == 0:
            self.count_content_words()

        # compute the vocabulary drift for each text in corpus 2 and append it as an 
        # additional column to corpus 2
        voc_drifts = []
        for i in range(len(self.corpus2)):
            voc_drift = 0
            for word in self.content_words:
                voc_drift += __compute_voc_drift(self.corpus2['text'][i])
            voc_drifts.append(voc_drift)

        self.corpus2['voc_drift'] = voc_drifts

        # compute the average vocabulary drift
        voc_drift = np.mean(voc_drifts)

        return voc_drift
    

    def __compute_voc_drift(self, text: str) -> float:
        """
        Compute the vocabulary drift between corpus 1 and a text in corpus 2,
        where the vocabulary drift is defined as cross-entropy between the
        word frequencies in corpus 1 and the word frequencies in the text in corpus 2.

        Parameters
        ----------
        text : A text in corpus 2.

        Returns
        -------
        voc_drift : The vocabulary drift between corpus 1 and the text in corpus 2.
        """
        # compute the word frequencies in corpus 1
        word_freqs1 = {}
        for word in self.content_words:
            word_freqs1[word] = self.content_word_counts[word][0] / len(self.corpus1)

        # compute the word frequencies in the text in corpus 2
        word_freqs2 = {}
        for word in self.content_words:
            word_freqs2[word] = text.count(word) / len(text)

        # compute the cross-entropy between the word frequencies in corpus 1 and the word frequencies in the text in corpus 2
        voc_drift = 0
        for word in self.content_words:
            voc_drift += word_freqs1[word] * np.log(word_freqs2[word])

        return voc_drift
    

    def return_sem_drift(self) -> float:
        """
        Return the average semantic drift between corpus 1 and the texts in corpus 2.

        Returns
        -------
        sem_drift : The average semantic drift between corpus 1 and the texts in corpus 2.
        """
        assert self.sbert_model != None, "You need to set the sbert_model attribute of the CorporaComparison object before you can compute the semantic drift."

        # compute the semantic drift for each text in corpus 2 and append it as an 
        # additional column to corpus 2
        sem_drifts = []
        for i in range(len(self.corpus2)):
            sem_drift = __compute_sem_drift()
            sem_drifts.append(sem_drift)

        self.corpus2['sem_drift'] = sem_drifts

        # compute the average semantic drift
        sem_drift = np.mean(sem_drifts)

        return sem_drift
    

    def __compute_sem_drift(self, text: str) -> float:
        """
        Compute the semantic drift between corpus 1 and a text in corpus 2,
        where the semantic drift is defined as the average cosine distance
        between the sbert embeddings of the content words that appear in both
        corpus 1 and the text.

        Parameters
        ----------
        text : A text in corpus 2.

        Returns
        -------
        sem_drift : The semantic drift between corpus 1 and the text in corpus 2.
        """
        # subset self.content_words to include only the content words that appear
        # in both corpus 1 and the text
        content_words = []
        for word in self.content_words:
            if word in text:
                content_words.append(word)

        # compute sentence-level sbert embeddings for the content words
        content_word_embeddings1 = self.sbert_model.encode(content_words)
        content_word_embeddings2 = self.sbert_model.encode(text)

        # compute the average cosine distance between the sbert embeddings of the content words that appear in both corpus 1 and the text
        sem_drift = 0
        for i in range(len(content_words)):
            sem_drift += __cos_dist(content_word_embeddings1[i], content_word_embeddings2[i])

        sem_drift /= len(content_words)

        return sem_drift
    

    def __cos_dist(self, vec1: np.array, vec2: np.array) -> float:
        """
        Compute the cosine distance between two vectors.

        Parameters
        ----------
        vec1 : A vector.
        vec2 : A vector.

        Returns
        -------
        cos_dist : The cosine distance between the two vectors.
        """
        cos_dist = 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        return cos_dist