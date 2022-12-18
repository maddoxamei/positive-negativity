import numpy as np
import os
import torch
import math

from word_vectors.transforms import *
from glob import glob
processor = TextProcessor('data/lexicons/NRC-VAD-Lexicon.txt', 'data/lexicons/glove.6B.50d.pickle')

def get_doc_sentences(path):
    with open(path, 'r') as file:
        return processor.get_sentences(file.read())

def get_embeddings(sentences):
    """

    :param document_index:
    :return:
    """
    embedded_sents = []
    for sent in sentences:
        embedding = processor.get_token_embeddings(sent, remove_pseudowords=True, remove_stopwords=True, polarization_thresh=(.5,.1,.1))
        if len(embedding) > 0:
            embedded_sents.append(np.asarray(embedding))
    return embedded_sents

def valence_extremity(embedded_sent):
    max_valence_index = np.argmax(embedded_sent[:, 0])
    return embedded_sent[max_valence_index]

def dimension_average(embedded_sent):
    return np.mean(embedded_sent, axis=0)

def get_sentence_vectors(embedded_sentences, reduction_function=dimension_average):
    """Get sentence-level metrics"""
    sentence_vectors = np.asarray([reduction_function(sent) for sent in embedded_sentences])
    # Look at both valence and arousal when determining sentiment
    sentence_vectors = sentence_vectors[:, 0]*sentence_vectors[:, 1]
    # Just look at valence when determining sentiment
    # sentence_vectors = sentence_vectors[:, 0]
    return sentence_vectors

def get_sections(sentence_vectors):
    four_fifths = int(len(sentence_vectors) * 4 / 5)
    return sentence_vectors[:four_fifths], sentence_vectors[four_fifths:]

def get_section_sentiment(section_average):
    # When looked at both valence and arousal when determining sentiment
    return bool(section_average > .5*.2)
    # When looked at valence when determining sentiment
    # return bool(section_average > .5)

def get_document_sentiment(sentence_vectors, threshold=.2):
    first_section, second_section = get_sections(sentence_vectors)

    # Entire document sentiment is simply the "first section" sentiment
    # if there is only one sentence in the entire document
    if len(second_section) == 0:
        return np.mean(first_section).round()

    # Get section averages
    first_section_average = np.mean(first_section)
    second_section_average = np.mean(second_section)

    # Increases weight of last sentence in longer documents
    if len(second_section) > 2:
        s1, s2 = get_sections(second_section)
        second_section_average = np.mean(s1)*np.mean(s2)

    first_section_sentiment = get_section_sentiment(first_section_average)
    second_section_sentiment = get_section_sentiment(second_section_average)

    # If the first and second sections have different sentiments, and
    #   the difference between the sentiments is sufficient, then
    #   Indicate there is the presence of thwarting.
    #   Otherwise, indicate the lack of thwarting
    different_sentiments = first_section_sentiment != second_section_sentiment
    thwarted = different_sentiments and np.abs(first_section_average-second_section_average) > threshold

    # If there is no thwarting, entire document sentiment is simply the "first section" sentiment
    # If there is IS thwarting, entire document sentiment is simply the "second section" sentiment
    return second_section_sentiment if thwarted else first_section_sentiment


documents = list(glob('data/IMDB_reviews/*.txt'))

strDir = r'C:\File\GitHub\positive-negativity'
for iter,doc_string in enumerate(documents):
    print(doc_string)
    sentences = get_doc_sentences(doc_string)
    embeddings = get_embeddings(sentences)
    sentence_vectors = get_sentence_vectors(embeddings)
    result = get_document_sentiment(sentence_vectors)
    # not sure what you were wanting to print and why
    np.savetxt(os.path.join(strDir, "data/rule_predictions",f"{doc_string}_{result}_sentence_metrics.csv", sentence_vectors))
    print(result)