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
    stop_words = stopwords.words('english')

    # extend the stop words with the following words
    stop_words.extend(['york', 'states', 'book', 'part', 'cambridge', 
                       'harvard', 'institution', 'press', 'even', 'princetion',
                       'united', 'known', 'chapter', 'chicago', 'brookings', 
                       'washington', 'oxford', 'paper', 'clarendon', 'hopkins',
                       'cent', 'wiley', 'chapters', 'publishing', 'would', 'first',
                       'american', 'review', 'bureau', 'per', 'em', 'great', 'years',
                       'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                        'nine', 'ten', 'zero', 'de', 'reviews', 'graduate', 'volume'])

    # return the stop words
    return stop_words


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
    return [word for word in words if not word.endswith('.') and not word.endswith(':') and not word.endswith(',')]


def remove_abbreviations_from_bigrams(bigrams: list) -> list:
    """
    Filter out abbreviations from a list of bigrams,
    where abbreviations are of the form 'word1 word2.',
    or 'word1 word2:' or 'word1 word2,' or 'word1. word2',
    or 'word1: word2' or 'word1, word2', or 'word1 word2&',
    or 'word1& word2'.

    Parameters
    ----------
    bigrams : The list of bigrams to filter abbreviations from.

    Returns
    -------
    bigrams_without_abbreviations : The list of bigrams without abbreviations.
    """
    # return the list of bigrams without abbreviations
    return [bigram for bigram in bigrams if not bigram[0].endswith('.') 
            and not bigram[0].endswith(':') and not bigram[0].endswith(',') 
            and not bigram[1].endswith('.') and not bigram[1].endswith(':') 
            and not bigram[1].endswith(',') and not bigram[0].endswith('&') 
            and not bigram[1].endswith('&') and not bigram[0].endswith(';')
            and not bigram[1].endswith(';')]


def remove_single_letter_bigrams(bigrams: list) -> list:
    """
    Remove bigrams where at least one word consists of only one character.

    Parameters
    ----------
    bigrams : The list of bigrams to remove single letter bigrams from.

    Returns
    -------
    bigrams_without_single_letter_bigrams : The list of bigrams without single letter bigrams.
    """
    # return the list of bigrams without single letter bigrams
    return [bigram for bigram in bigrams if len(bigram[0]) > 1 and len(bigram[1]) > 1]


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


def convert_bigrams_to_unigrams(bigrams: list) -> list:
    """
    Given a list of bigrams, convert the bigrams to unigrams.
    Include a space between the two words in the bigram if
    there is no space between them already.

    Parameters
    ----------
    bigrams : The list of bigrams to convert to unigrams.

    Returns
    -------
    unigrams : The list of unigrams.
    """
    # return the list of unigrams
    return [bigram[0] + ' ' + bigram[1] if bigram[0][-1] != ' ' else bigram[0] + bigram[1] for bigram in bigrams]


def collect_unigrams(list_of_unigrams_lists: list) -> list:
    """
    Given a list of lists of unigrams, collect the unigrams
    into a single list and remove duplicates.

    Parameters
    ----------
    list_of_unigrams_lists : The list of lists of unigrams to collect.

    Returns
    -------
    unigrams : The list of unigrams.
    """
    # return the list of unigrams
    return list(set([unigram for unigrams_list in list_of_unigrams_lists for unigram in unigrams_list]))


def get_content_words(documents: list) -> list:
    """
    Given a list of documents, get the content words from the documents.

    Parameters
    ----------
    documents : The list of documents to get the content words from.

    Returns
    -------
    content_words : The list of content words.
    """
    stop_words = get_nlkt_stop_words()

    # convert documents to lowercase and remove line breaks
    documents = remove_line_breaks(docs_to_lowercase(documents))

    # find the most common unigrams in each document
    unigrams = []
    num_unigrams = 30

    for document in documents:
        raw_unigrams = find_common_unigrams(document, num_unigrams)

        # filter unigrams
        unigrams.append(remove_numerical_words(remove_abbreviations(filter_unigrams(raw_unigrams, stop_words))))

    # find the most common bigrams in each document
    bigrams = []
    num_bigrams = 70

    for document in documents:
        raw_bigrams = find_common_bigrams(document, num_bigrams)

        # filter bigrams
        bigrams.append(remove_single_letter_bigrams(remove_universities_from_bigrams(remove_abbreviations_from_bigrams(remove_stopwords_from_bigrams(raw_bigrams, stop_words)))))

    # convert bigrams to unigrams
    unigrams_from_bigrams = []

    for bigrams_list in bigrams:
        unigrams_from_bigrams.append(convert_bigrams_to_unigrams(bigrams_list))

    # add unigrams_from_bigrams[i] to unigrams[i]
    for i in range(len(unigrams)):
        unigrams[i] += unigrams_from_bigrams[i]
    
    # collect unigrams
    unigrams = collect_unigrams(unigrams)

    # collect unigrams from bigrams
    unigrams_from_bigrams = collect_unigrams(unigrams_from_bigrams)

    return unigrams + unigrams_from_bigrams



# create a function that takes in two lists with entries (mean, standard deviation) and
# create a line plot with the means and standard deviations

def plot_mean_and_standard_deviation(data_1: list, data_2: list, x_axis_labels: list):
    """
    Given two lists of data of the form [(mean, standard deviation), ...],
    plot the means as two lines, one for each list, and plot the standard deviations
    as boxes around the lines.

    Parameters
    ----------
    data_1 : The first list of data.

    data_2 : The second list of data.

    x_axis_labels : The labels for the x-axis.
    """
    # import packages
    import matplotlib.pyplot as plt

    # create the figure
    plt.figure(figsize=(10, 5))

    # plot the means
    plt.plot(data_1, label='BERT')
    plt.plot(data_2, label='SBERT')

    # plot the standard deviations as error bars around the means
    plt.errorbar(range(len(data_1)), [data[0] for data in data_1], [data[1] for data in data_1], linestyle='None', marker='^', color='blue')
    plt.errorbar(range(len(data_2)), [data[0] for data in data_2], [data[1] for data in data_2], linestyle='None', marker='^', color='orange')

    # set the x-axis labels
    plt.xticks(range(len(x_axis_labels)), x_axis_labels)

    # set the title and the legend
    plt.title('Mean and standard deviation of the cosine similarity between the embeddings of the content words')
    plt.legend()

    # show the plot
    plt.show()


def substring_matching(content_words: set) -> set:
    """
    For each content word, check if it is a substring of another content word.
    If so, remove the content word from the set of content words.
    """
    # create a list of the content words
    content_words_list = list(content_words)

    # for each content word (use tqdm)
    for i in tqdm(range(len(content_words_list))):
        # for each content word
        for j in range(len(content_words_list)):
            # if the content word is a substring of another content word
            if i != j and content_words_list[i] in content_words_list[j]:
                # remove the content word from the set of content words
                content_words.remove(content_words_list[i])

    return content_words


def create_frequency_hist(sorted_content_words: list) -> None:
    """
    Given a list of content words with entries of the form
    (content word, frequency), create a histogram of the frequencies.
    On the y-axis, plot the percentage of content words with a given frequency.
    Use a logarithmic scale for the y-axis.
    """
    # import packages
    import matplotlib.pyplot as plt

    # create the figure
    plt.figure(figsize=(10, 5))

    # create the histogram
    plt.hist([content_word[1] for content_word in sorted_content_words], bins=100, density=True, cumulative=True, histtype='step')

    # set the title and the labels
    plt.title('Frequency of content words')
    plt.xlabel('Frequency')
    plt.ylabel('Percentage of content words')

    # set the y-axis to a logarithmic scale
    plt.yscale('log')

    # show the plot
    plt.show()


# count the number of appearances of an input string in a larger string
# the input string need not be a single word

def count_substring_matches(input_string: str, text: str) -> int:
    """
    Given an input string and a larger string, count the number of appearances
    of the input string in the larger string.
    """
    # split the input string into words
    input_string_split = input_string.split()

    # count the number of appearances of the input string in the larger string
    count = 0
    for i in range(len(text.split()) - len(input_string_split) + 1):
        if text.split()[i:i+len(input_string_split)] == input_string_split:
            count += 1

    return count


# given a list of integers, create a line plot of the integers
# the x-axis should be the indices of the integers + 1

def plot_integers(integers: list) -> None:
    """
    Given a list of integers, create a line plot of the integers.
    The x-axis should be the indices of the integers + 1.
    """
    # import packages
    import matplotlib.pyplot as plt

    # create the figure
    plt.figure(figsize=(10, 5))

    # create the line plot
    plt.plot(range(1, len(integers) + 1), integers)

    # set the title and the labels
    plt.title('Number of content words with a given frequency')
    plt.xlabel('Frequency')
    plt.ylabel('Number of content words')

    # start the y-axis at zero
    plt.ylim(bottom=0)

    # show the plot
    plt.show()


def get_n_gram_dict(list_of_strings: list) -> dict:
    """
    Given a list of strings, return a dictionary with the integers from 1
    to the length of the longest n-gram as keys, and the number of n-grams
    of length i as values.
    """
    # create a dictionary with the integers from 1 to the length of the longest n-gram as keys
    # and the number of n-grams of length i as values
    n_gram_dict = {}

    # for each string in the list of strings
    for string in list_of_strings:
        # for each n-gram in the string
        for i in range(1, len(string.split()) + 1):
            # if the length of the n-gram is not in the dictionary, add it
            if i not in n_gram_dict:
                n_gram_dict[i] = 1
            # if the length of the n-gram is in the dictionary, increase its count by 1
            else:
                n_gram_dict[i] += 1

    return n_gram_dict