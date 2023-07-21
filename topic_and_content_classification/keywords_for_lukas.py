from tqdm import tqdm
from math import log
import numpy as np
import pandas as pd
from itertools import combinations

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def tfidf_keywords(path: str, bigrams: bool=False, n: int=40) -> list:
    """
    Mines a test set for optimal keywords, using TF-IDF algorithm 

    Parameters
    ----------
    path : Path to training data. This should be a csv, with columns 'article' and 'label'
    bigrams : If true, returns bigram key-phrases. Else returns single words
    n : number of keywords to return 

    Returns
    ----------
    n keywords, sorted ordered from best to worst
    """
    
    # Open data
    train = pd.read_csv(path)
    train_arts = list(train['article'])
    train_labs = list(train['label'])

    on_topic = [train_arts[i] for i in range(len(train_arts)) if train_labs[i] == 1]
    on_topic = " ".join(on_topic)
    off_topic = [train_arts[i] for i in range(len(train_arts)) if train_labs[i] == 0]
    off_topic = " ".join(off_topic)

    # Preprocessing 
    on_topic = on_topic.lower()
    off_topic = off_topic.lower()

    # Remove stopwords
    # nltk.download('stopwords')
    # nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    on_top_word_tokens = word_tokenize(on_topic)
    on_top_word_tokens = [w for w in on_top_word_tokens if not w in stop_words]
    off_top_word_tokens = word_tokenize(off_topic)
    off_top_word_tokens = [w for w in off_top_word_tokens if not w in stop_words]

    # Remove punctuation 
    symbols = "!\"#$%&()*+-.,/:;<=>?@[\]'^_`{|}~\n"

    no_punct_on_top_word_tokens = []
    for word in on_top_word_tokens:
        for i in symbols:
            word = word.replace(i, '')
        if len(word) > 0:
            no_punct_on_top_word_tokens.append(word)

    no_punct_off_top_word_tokens = []
    for word in off_top_word_tokens:
        for i in symbols:
            word = word.replace(i, '')
        if len(word) > 0:
            no_punct_off_top_word_tokens.append(word)

    # Convert to bigrams if desired
    if bigrams:
        no_punct_on_top_word_tokens = [" ".join(b) for b in nltk.bigrams(no_punct_on_top_word_tokens)]
        no_punct_off_top_word_tokens = [" ".join(b) for b in nltk.bigrams(no_punct_off_top_word_tokens)]

    print(len(no_punct_on_top_word_tokens), "on topic words")
    print(len(no_punct_off_top_word_tokens), "off topic words")

    # Document frequency
    DF = {}
    for word in no_punct_on_top_word_tokens + no_punct_off_top_word_tokens:
        if word not in DF:
            DF[word] = 0
        DF[word] += 1

    N = len(DF)
    print(f'{N} unique words')

    # Inverse document frequency
    IDF = {}
    for word in DF:
        IDF[word] = log(N/(DF[word] + 1))

    # Term frequency
    TF = {}
    for word in no_punct_on_top_word_tokens:
        if word not in TF:
            TF[word] = 0
        TF[word] += 1

    for word in TF:
        TF[word] = TF[word]/len(no_punct_on_top_word_tokens)

    # TFIDF
    TFIDF = {}
    for word in TF:
        TFIDF[word] = TF[word] * IDF[word]

    # Return top n
    sorted_vals = sorted(list(TFIDF.values()), reverse=True)
       
    max_keys = [key for key, value in TFIDF.items() if value >= sorted_vals[n-1]]
    values = [TFIDF[key] for key in max_keys]

    sorted_max_keys = [x for _, x in sorted(zip(values, max_keys), reverse=True)]

    return sorted_max_keys


def best_kws_from_list(path: str, list_of_kws: list, sequential: bool=True) -> list:
    """
    From a list of keywords, finds the sublist that optimises F1 

    Parameters
    ----------
    path : Path to labelled data. This should be a csv, with columns 'article' and 'label'
    list_of_kws: List of keywords to test. 
    sequential : If true additively finds the best keywords. So the best 1 keyword will always
        be one the best best 2 keywords. If false, considers all combinations. But this is 
        pretty quickly computationally infeasible. 

    Returns
    ----------
    List of keywords that maximises F1
    """

    # Open data
    dat = pd.read_csv(path)
    arts = list(dat['article'])
    gt_labels = np.array(list(dat['label']))

    # Iterate through keywords 
    best_f1 = 0
    best_kws = []
    new_best_kws = []

    for i in range(1, len(list_of_kws) + 1):
        print(f'Finding keyword {i}')

        if sequential:
            for kw in list_of_kws:
                kw_combo = best_kws + [kw]

                pred_labels = np.array([any(kw in art for kw in kw_combo) for art in arts]).astype(int)
                F1 = metrics(gt_labels, pred_labels)

                if F1 > best_f1:
                    best_f1 = F1
                    new_best_kws = kw_combo
        
        else:
            for kw_combo in tqdm(combinations(list_of_kws, i)):

                pred_labels = np.array([any(kw in art for kw in kw_combo) for art in arts]).astype(int)
                F1 = metrics(gt_labels, pred_labels)

                if F1 > best_f1:
                    best_f1 = F1
                    new_best_kws = kw_combo

        best_kws = new_best_kws
    
        print(f"Best F1: {best_f1}")
        print(f"Best {i} keywords: {best_kws}")

    return best_kws


def metrics(gt_labels: np.array, pred_labels: np.array) -> float:
    """
    Given predicted labels and ground truth labels, evaluate performance. 
    Print recall, precision, F1, accuracy.
    ----------
    gt_labels : numpy array of ground truth labels. 
    pred_labels : numpy array of predicted labels. 

    Returns
    ----------
    F1
    """

    tps = np.sum((pred_labels == 1) & (gt_labels == 1))
    fps = np.sum((pred_labels == 1) & (gt_labels == 0))
    tns = np.sum((pred_labels == 0) & (gt_labels == 0))
    fns = np.sum((pred_labels == 0) & (gt_labels == 1))

    if tps+fns > 0:
        recall = tps/(tps+fns)
    else:
        recall = 0
    print("Recall:", round(recall*100, 1))
    if tps + fps > 0:
        precision = tps/(tps + fps)
    else:
        precision = 0
    print("Precision:", round(precision*100, 1))
    if precision + recall > 0:
        F1 = 2 * (precision * recall)/(precision + recall)
    else:
        F1 = 0
    print("F1:", round(F1*100, 1))
    accuracy = (tps + tns)/len(pred_labels)
    print("Accuracy:", round(accuracy*100, 1))

    return F1


def evaluate(path: str, kw_list: list) -> None:
    """
    Evaluates set of keywords on labelled dataset.  
    Prints recall, precision, F1, accuracy.
    ----------
    path : Path to labelled data. This should be a csv, with columns 'article' and 'label'
    kw_list : List of keywords
    """

    # Open data
    dat = pd.read_csv(path)
    arts = list(dat['article'])
    gt_labels = np.array(list(dat['label']))

    # Predict
    pred_labels = np.array([any(kw in art for kw in kw_list) for art in arts]).astype(int)

    # Evaluate    
    metrics(gt_labels, pred_labels)


if __name__ == '__main__':

    root = '/mnt/data01/topic/finetuning/train_sets/ww1/'

    # Get lists of suggested keywords
    tfidf_kws = tfidf_keywords(path=f'{root}/train.csv', bigrams=False, n=40)
    tfidf_bgs = tfidf_keywords(path=f'{root}/train.csv', bigrams=True, n=40)

    # Find optimal subset
    best_kws = best_kws_from_list(
        path=f'{root}/dev.csv', 
        list_of_kws=tfidf_kws+tfidf_bgs, 
        sequential=True
    )

    # Evaluate on test set
    evaluate(path=f'{root}/test.csv', kw_list=best_kws)
