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
    def __init__(self, vad_lexicon_file: str, glove_searchspace_file: str, k: int = 10, **kwargs) -> None:
        self.k = k

        logging.info("Creating a vector database of the NRC Valence-Arousal-Dominance Lexicon")
        vad = np.loadtxt(vad_lexicon_file, dtype=str, delimiter='\t')
        self.vad_lexicon = dict(zip(vad[:, 0], vad[:, 1:].astype(np.float32)))

        logging.info("Creating a vector database of the Global Vectors for Word Representation Lexicon")
        with open(glove_searchspace_file, 'rb') as file:
            self.glove_searchspace = pickle.load(file)

        self.nlp = spacy.load("en_core_web_lg", exclude=['ner'])
        customize_stopwords(self.nlp, [' '])
        self.nlp.add_pipe('sentence_punctuation', before='parser')
        self.nlp.add_pipe('clausing', before='parser')
        self.nlp.add_pipe('pseudo_words', after='lemmatizer')
        self.nlp.add_pipe('negation', last=True)

        spacy.tokens.Token.set_extension("is_pseudo", default=False, force=True)
        spacy.tokens.Token.set_extension("is_negated", default=False, force=True)
        spacy.tokens.Token.set_extension("vad_vector", getter=self._get_vad_representation, force=True)
        spacy.tokens.Span.set_extension("vad_vector", getter=self._get_filled_vectors, force=True)
        spacy.tokens.Doc.set_extension("vad_vector", getter=self._get_filled_vectors, force=True)

    def __call__(self, text: str) -> List[str]:
        """
        :param clause:
        :return: words separated
        """
        text = text.lower()
        text = self._space_formatting(text)
        return self.nlp(text)

    def get_sentences(self, obj: Union[str, spacy.tokens.Doc, spacy.tokens.Span]) -> List[spacy.tokens.span.Span]:
        if isinstance(obj, str):
            obj = self.__call__(obj)
        return list(obj.sents)

    def get_token_obj(self, obj: Union[str, spacy.tokens.Doc, spacy.tokens.Span], **kwargs) -> List[spacy.tokens.span.Span]:
        if isinstance(obj, str):
            obj = self.__call__(obj)
        return [token for token in self._filter_tokens(obj, **kwargs)]

    def get_token_strings(self, obj: Union[str, spacy.tokens.Doc, spacy.tokens.Span], **kwargs) -> List[
        spacy.tokens.span.Span]:
        return [token.lemma_ for token in self.get_token_obj(obj, **kwargs)]

    def get_token_embeddings(self, obj: Union[str, spacy.tokens.Doc, spacy.tokens.Span], **kwargs) -> List[
        spacy.tokens.span.Span]:
        return [token._.vad_vector for token in self.get_token_obj(obj, **kwargs)]

    def _filter_tokens(self, obj: Union[spacy.tokens.Doc, spacy.tokens.Span], remove_pseudowords: bool = False,
                       remove_stopwords: bool = True, polarization_thresh: Tuple[float, float, float] = (0, 0, 0),
                       neutral: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> List[spacy.tokens.Token]:
        """ Return non-punctuation tokens __ with pseudowords and stopwords removed if desired

        :param obj:
        :param remove_pseudowords:
        :param remove_stopwords:
        :param polarization_thresh: how-far away from "neutral" (.5) a dimension must be to include the word in the sentence
        :return:
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
        vector = self.vad_lexicon.get(token.lemma_.lower(), np.array([np.nan, np.nan, np.nan]))
        if token._.is_negated:
            return np.array([1-vector[0], vector[1], vector[2]])
        return vector

    def _get_filled_vectors(self, obj: Union[spacy.tokens.Doc, spacy.tokens.Span]):
        vectors = np.asarray([token._.get('vad_vector') for token in obj])
        vectors = np.where(np.isnan(vectors), np.nanmean(vectors, axis=0), vectors)
        return np.nan_to_num(vectors)
        # vectors = []
        # for token in obj:
        #     vector = token._.get('vad_vector')
        #     if vector is None:
        #         vector = self._handle_unknowns(token)
        #     vectors.append(vector)
        # vectors = np.asarray(vectors)
        # vectors = np.where(np.isnan(vectors), np.nanmean(vectors, axis=0), vectors)
        # return np.nan_to_num(vectors)

    def _handle_unknowns(self, token: spacy.tokens.Token) -> np.ndarray:
        """
        Assumes the original token is not present in the VAD database

        Returns NaN if:
            - original token does not exist in GLoVe
            - original token nor any of the k-nearest words exist in VAD
        :param token:
        :return:
        """
        # Token does not exist in VAD, but does in GLoVe
        if token.has_vector:
            return self.get_k_nearest(token.vector)
        logging.info(f"Token <{token}> has no GLoVe representation. Representation will be computed from all other tokens in the clause...")
        return np.array([np.nan, np.nan, np.nan])

    def get_k_nearest(self, encoding):
        """
        :param encoding:
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