import pickle
from typing import List

from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize

# from flair.data import Sentence
# from flair.models import TextClassifier

import numpy as np

def get_doc_sentences(path, processor) -> List["spacy.tokens.Span"]:
    with open(path, 'r') as file:
        return processor.get_sentences(file.read())

# def get_sentence_sentiments_from_pretrained(sentences, processor, binarized=False):
#     classifier = TextClassifier.load('en-sentiment')
#     sentiments = []
#     for sent in sentences:
#         tokens = processor.get_token_strings(sent, remove_pseudowords=False, remove_stopwords=False)
#         if len(tokens) == 0:
#             continue
#         phrase = Sentence(' '.join(tokens))
#         classifier.predict(phrase)
#         results = phrase.labels[0].to_dict()
#         binary_label = int(results.get('value')=='POSITIVE')
#         if binarized:
#             sentiments.append(binary_label)
#         else:
#             sentiments.append(np.interp(results.get('confidence'), [0, 1], [.5, binary_label]))
#     return np.array(sentiments)


def get_embeddings(sentences, processor: "word_vectors.transforms.TextProcessor") -> List[np.ndarray]:
    """

    :param sentences:
    :param processor:
    :return: list of sentence embeddings of shape (NUM_OF_WORDS_IN_SENTENCE, 3);
            length of list is equivalent to NUM_OF_SENTENCES
    """
    embedded_sents = []
    for sent in sentences:
        embedding = processor.get_token_embeddings(sent, remove_pseudowords=True, remove_stopwords=True, polarization_thresh=(.2,.1,.1))
        if len(embedding) > 0:
            embedded_sents.append(np.asarray(embedding))
    return embedded_sents


def valence_extremity(embedded_sent, neutral: float = 0.5):
    max_valence_index = np.argmax(np.abs(embedded_sent[:, 0]-neutral))
    return embedded_sent[max_valence_index]


def dimension_average(embedded_sent):
    return np.mean(embedded_sent, axis=0)


def get_sentence_vectors(embedded_sentences: List[np.ndarray], reduction_function=dimension_average, valence_only: bool = False) -> np.ndarray:
    """ Represent sentences as a single number

    :param embedded_sentences:
    :param reduction_function: how to obtain single-number from a sentence of words
    :return: array of shape (NUM_OF_SENTENCES, )
    """
    sentence_vectors = np.asarray([reduction_function(sent) for sent in embedded_sentences])
    # Just look at valence when determining sentiment
    if valence_only:
        sentence_vectors = sentence_vectors[:, 0]
    # Look at both valence and arousal when determining sentiment
    else:
        sentence_vectors = sentence_vectors[:, 0]*sentence_vectors[:, 1]
    return sentence_vectors


def create_glove_searchspace(labels, encodings):
    # Row-wise normalization to make cosine distance translate to euclidean distance
    encodings = normalize(encodings, axis=1, norm='l2')
    tree = BallTree(encodings, metric='l2')
    with open('data/lexicons/glove.6B.300d_l2.pickle', 'wb') as file:
        pickle.dump({'labels': np.asarray(labels), 'tree': tree}, file)

