from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def compute_cosine_similarity(matrix):
    cosine_sim = cosine_similarity(matrix)
    return cosine_sim[0, 1]

def label_encode(tokens, all_tokens):
    label_encoder = LabelEncoder()
    label_encoder.fit(all_tokens)
    encoded_labels = label_encoder.transform(tokens)
    return encoded_labels

def one_hot_encode(tokens, all_tokens):
    one_hot_encoder = OneHotEncoder(sparse_output=False, categories=[sorted(set(all_tokens))])
    one_hot_encoded = one_hot_encoder.fit_transform(np.array(tokens).reshape(-1, 1))
    return one_hot_encoded

def bag_of_words(documents):
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(documents)
    return X_bow

def tfidf(documents):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(documents)
    return X_tfidf
