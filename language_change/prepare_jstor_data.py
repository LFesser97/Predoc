import json
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from glob import glob
import pickle
from collections import Counter

# import nltk and download stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# Preparing Jstor json into a df

def process_data_json(data_json_path):
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
    # jstor_df['abstract_stripped'] = jstor_df['abstract'].str.strip()
    #jstor_df['fulltext'] = jstor_df['fulltext'].str.strip()

    return jstor_df


# Read a json df's column 'abstract' into a list of documents

def get_documents_from_path(data_json_path):
    jstor_df = process_data_json(data_json_path)
    documents = []
    for fulltext in jstor_df['fulltext']:
        documents.append(fulltext)

    return documents


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


def get_docs(path: str, year: int) -> list:
    """
    Given a path to a json file and a year, returns a list of documents from that year.

    Parameters
    ----------
    path : A path to a json file.

    year : An integer representing a year.

    Returns
    ----------
    documents : A list of documents from the given year.
    """
    documents = get_documents_from_path(path)
    jstor_df = process_data_json(path)
    baseline_df = jstor_df[jstor_df['year'] == year]
    baseline_docs_raw = [baseline_df.iloc[i]['fulltext'] for i in range(len(baseline_df))]

    documents = preprocess_docs(baseline_docs_raw)

    return documents


def get_dict_of_pairs(baseline_docs_with_title: list, content_words: list) -> dict:
    """
    Given a list of documents and a list of content words, returns a dictionary
    with content words as keys and a list of ten tuples. Each tuple contains
    two documents from baseline_docs that contain the content word in the key.

    Parameters
    ----------
    baseline_docs : A list of documents.

    content_words : A list of content words.

    Returns
    ----------
    dict_of_pairs : A dictionary with content words as keys and a list of ten tuples.
    """
    dict_of_pairs = {}
    for word in content_words:
        dict_of_pairs[word] = []

    for word in content_words:
        # get indices of documents that contain word
        indices = [i for i, doc in enumerate(baseline_docs_with_title) if word in doc[0]]

        # get all pairs of documents that contain word
        pairs = [[{'title': baseline_docs_with_title[i][1], 'text': baseline_docs_with_title[i][0]},
            {'title': baseline_docs_with_title[j][1], 'text': baseline_docs_with_title[j][0]}] for i in indices for j in indices if i < j]

        # get ten random pairs
        random_pairs = []
        for i in range(10):
            random_pairs.append(pairs.pop(np.random.randint(len(pairs))))

        # for each tuple in random_pairs, and for each document in the tuple,
        # add only the first time the content word appears in the document
        # with 64 words on each side of the content word
        for pair in random_pairs:
            for i in range(2):
                doc = pair[i]['text']
                if word in doc:
                    index = doc.index(word)
                    doc = doc[index - 64:index + 64]
                    pair[i]['text'] = doc

        dict_of_pairs[word] = random_pairs

    return dict_of_pairs