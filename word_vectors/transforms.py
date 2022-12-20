import logging
import pickle
from typing import List, Union, Generator, Tuple
import re

import numpy as np
import spacy.tokens
from sklearn.neighbors import BallTree
from word_vectors.spacy_utils import *


class TextProcessor(object):
    """Preprocess data and handle edge-cases present in the training/test corpus.

        Cleanup process does NOT include token/word standardization."""
    def __init__(self, vad_lexicon_file: str, glove_searchspace_file: str, k: int = 10, clauses_parsing: bool = True, **kwargs) -> None:
        """
        :param vad_lexicon_file: Local path (folder+filename+extension) to the NRC-VAD Lexicon text file
        :param glove_searchspace_file: Local path (folder+filename+extension) to the serialized GLoVe search space object
        :param k: how many symantically similar words to check VAD vector representations for
        :param clauses_parsing: whether to split text into clauses; if False, splits text into sentences
        :param kwargs:
        """
        self.k = k

        logging.info("Creating a vector database of the NRC Valence-Arousal-Dominance Lexicon")
        vad = np.loadtxt(vad_lexicon_file, dtype=str, delimiter='\t')
        self.vad_lexicon = dict(zip(vad[:, 0], vad[:, 1:].astype(np.float32)))

        logging.info("Creating a vector database of the Global Vectors for Word Representation Lexicon")
        with open(glove_searchspace_file, 'rb') as file:
            self.glove_searchspace = pickle.load(file)

        # Establish text processing pipeline
        self.nlp = spacy.load("en_core_web_sm", exclude=['ner'])
        customize_stopwords(self.nlp, [' '])
        self.nlp.add_pipe('sentence_punctuation', before='parser')
        if clauses_parsing:
            self.nlp.add_pipe('clausing', before='parser')
        self.nlp.add_pipe('pseudo_words', after='lemmatizer')
        self.nlp.add_pipe('negation', last=True)

        # Add attributes to the spacy objects for later use
        spacy.tokens.Token.set_extension("is_pseudo", default=False, force=True)
        spacy.tokens.Token.set_extension("is_negated", default=False, force=True)
        spacy.tokens.Token.set_extension("vad_vector", getter=self._get_vad_representation, force=True)
        spacy.tokens.Span.set_extension("vad_vector", getter=self.get_token_embeddings, force=True)
        spacy.tokens.Doc.set_extension("vad_vector", getter=self.get_token_embeddings, force=True)

    def __call__(self, text: str) -> spacy.tokens.Doc:
        """
        :param text: desired text to preprocess
        :return:
        """
        text = text.lower()
        text = self._space_formatting(text)
        return self.nlp(text)

    def get_sentences(self, obj: Union[str, spacy.tokens.Doc, spacy.tokens.Span]) -> List[spacy.tokens.span.Span]:
        """ Split text into sentences

        :param obj: raw text or preprocessed text
        :return: list of sentences
        """
        # Apply preprocessing if not already done so
        if isinstance(obj, str):
            obj = self.__call__(obj)
        # Split text into sentences
        return list(obj.sents)

    def get_token_obj(self, obj: Union[str, spacy.tokens.Doc, spacy.tokens.Span], **kwargs) -> List[spacy.tokens.span.Span]:
        """ Retrieves token objects from a collection of text. Applies filtering as indicated.

        :param obj: raw text or preprocessed text
        :param kwargs: non-keyword arguments for _filter_tokens function
        :return: filtered text
        """
        # Apply preprocessing if not already done so
        if isinstance(obj, str):
            obj = self.__call__(obj)
        # Get tokens based on filtration criteria
        return [token for token in self._filter_tokens(obj, **kwargs)]

    def get_token_strings(self, obj: Union[str, spacy.tokens.Doc, spacy.tokens.Span], **kwargs) -> List[str]:
        """ Retrieves words (as strings) from a collection of text. Applies filtering as indicated.

        :param obj: raw text or preprocessed text
        :param kwargs: non-keyword arguments for _filter_tokens function
        :return: filtered text
        """
        return [token.lemma_ for token in self.get_token_obj(obj, **kwargs)]

    def get_token_embeddings(self, obj: Union[str, spacy.tokens.Doc, spacy.tokens.Span], **kwargs) -> List[np.ndarray]:
        """ Retrieves words (as vectors) from a collection of text. Applies filtering as indicated.

        If word does not exist in the VAD lexicon, the k most syntatically-similar words within the GLoVe lexicon are used.
        If GLoVe has no vector representation for a particular word or the k most similar are not present in the VAD lexicon, the token's VAD values are imputed with the clause-level averages.
        If all words in a clause are unknown after the above, simply ignore the entire clause/sentence

        :param obj: spacy-preprocessed text object
        :param kwargs: non-keyword arguments for _filter_tokens function
        :return: vector representation of filtered text
        """
        # Simple retrieval, impute mean if not present
        # vectors = np.asarray([token._.get('vad_vector') for token in obj])
        # vectors = np.where(np.isnan(vectors), np.nanmean(vectors, axis=0), vectors)
        # return np.nan_to_num(vectors)
        vectors = []
        for token in self.get_token_obj(obj, **kwargs):
            vector = token._.get('vad_vector')
            # Word not exist in VAD lexicon
            if vector is None:
                vector = self._handle_unknowns(token)
                # Apply negation. Due to location, negation does not occur on mean-imputed word vectors
                if token._.is_negated:
                    vector = np.array([1 - vector[0], vector[1], vector[2]])
            vectors.append(vector)
        vectors = np.asarray(vectors)
        # Case to handle no-word clause/sentences after filtration
        #   (i.e. all composed of stop words, punctuation, not polarized enough words, etc.)
        if len(vectors) == 0:
            return vectors
        # Remaining unknown tokens imputed with mean
        vectors = np.where(np.isnan(vectors), np.nanmean(vectors, axis=0), vectors)
        # Represent ignored clause/sentence as an empty matrix
        return vectors[~np.isnan(vectors[:,0])]
        # Represent ignored clause/sentence as a zeroed matrix
        return np.nan_to_num(vectors)

    def _filter_tokens(self, obj: Union[spacy.tokens.Doc, spacy.tokens.Span], remove_pseudowords: bool = False,
                       remove_stopwords: bool = True, polarization_thresh: Tuple[float, float, float] = (0, 0, 0),
                       neutral: Tuple[float, float, float] = (0.5, 0.0, 0.0)) -> List[spacy.tokens.Token]:
        """ Return non-punctuation tokens __ with pseudowords and stopwords removed if desired

        :param obj: desired text to filter
        :param remove_pseudowords: whether to remove pseudowords from the given text
        :param remove_stopwords: whether to remove stop words from the given text
        :param polarization_thresh: how-far away from "neutral" a dimension must be to include the word in the sentence
        :return: filtered text
        """
        if isinstance(obj, str):
            obj = self.__call__(obj)
        polarization_thresh = np.asarray(polarization_thresh)
        neutral = np.asarray(neutral)
        return [token for token in obj if
                not token.is_punct and not (token._.is_pseudo and remove_pseudowords) and not (
                            token.is_stop and remove_stopwords) and np.all(np.abs(token._.vad_vector-neutral) >= polarization_thresh)]

    def _space_formatting(self, text: str) -> str:
        # substitute newline character and "p"/"b" tags with space
        text = re.sub(r'<\s*[/]?\s*p>|<\s*[/]?\s*b>|\n', ' ', text)
        # Separate num. from letters (e.g. "2PM" to "2 PM")
        text = re.sub(r'(\d)(\D)', r'\1 \2', text)
        return text

    def _get_vad_representation(self, token: spacy.tokens.Token):
        """ Get the VAD vector of a particular token, negating if applicable.

        :param token: desired token to obtain the VAD representation for
        :return: word vector if present in VAD lexicon, otherwise nan vector
        """
        vector = self.vad_lexicon.get(token.lemma_.lower(), np.array([np.nan, np.nan, np.nan]))
        if token._.is_negated:
            return np.array([1-vector[0], vector[1], vector[2]])
        return vector

    def _handle_unknowns(self, token: spacy.tokens.Token) -> np.ndarray:
        """
        Assumes the original token is not present in the VAD database

        Returns NaN if:
            - original token does not exist in GLoVe
            - original token nor any of the k-nearest words exist in VAD
        Otherwise, returns the VAD vector of the most similar token
        """
        # Token does not exist in VAD, but does in GLoVe
        if token.has_vector:
            location = self.glove_searchspace.get('labels').index(token.lemma_.lower())
            return self.get_k_nearest(self.glove_searchspace.get('tree').data.base[location])
        logging.info(f"Token <{token}> has no GLoVe representation. Representation will be computed from all other tokens in the clause...")
        return np.array([np.nan, np.nan, np.nan])

    def get_k_nearest(self, encoding):
        """ Finds the most-similar token
        :param encoding: GLoVe vector of the original token which is missing from the VAD lexicon
        :return: most similar vad-vector within the closest k, if applicable
        """
        # Find 5 most symantically-similar words to encoding
        dist, ndxs = self.glove_searchspace.get('tree').query(encoding.reshape(1, -1), k=self.k)
        for i in ndxs[0]:
            word_string = self.glove_searchspace.get('labels')[i]
            encoding = self.vad_lexicon.get(word_string)
            if encoding is not None:
                return encoding
        logging.info(f"Token <{word_string}> has no LAD-represented words within the {self.k} symantically-closest words.")
        return np.array([np.nan, np.nan, np.nan])