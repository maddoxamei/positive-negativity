import re
from typing import List, Union, Set, NoReturn

import nltk
import torchtext
from sklearn.neighbors import KDTree
from stopwords import get_stopwords
import numpy as np
import logging


class WordVectorEncoder(object):
    """
    NRC Valence, Arousal, Dominance (VAD) lexicon
    GloVe: Global Vectors for Word Representation
    """
    def __init__(self, vad_lexicon_file: str, glove_lexicon_file: str, k: int = 5, **kwargs) -> NoReturn:
        """
        :param unknown_token: token for unknown words which may be encountered during testing
        """
        self.k = k

        logging.info("Creating a vector database of the NRC Valence-Arousal-Dominance Lexicon")
        vad = np.loadtxt(vad_lexicon_file, dtype=str, delimiter='\t')
        vectors = vad[:, 1:].astype(np.float64)
        self.vad_lexicon = dict(zip(vad[:, 0], vectors))
        self.vad_searchspace = {'labels': vad[:, 0], 'tree': KDTree(vectors, metric="euclidean")}

        logging.info("Creating a vector database of the Global Vectors for Word Representation Lexicon")
        with open(glove_lexicon_file, 'r', encoding='utf8') as file:
            words = []
            vectors = []
            for line in file.readlines():
                word, *vector = line.split()
                words.append(word)
                vectors.append(vector)
            vectors = np.asarray(vectors).astype(np.float32)
        self.glove_lexicon = dict(zip(words, vectors))
        self.glove_searchspace = {'labels': words, 'tree': KDTree(vectors, metric="euclidean")}

    def __call__(self, tokens: List[str]):
        return self.transform(tokens)

    def transform(self, tokens: List[str]):
        """
        Imputes missing "vectors" (words) with the mean vector representation per column, when applicable
        :param tokens:
        :return:
        """
        vectors = []
        for token in tokens:
            encoding = self.vad_lexicon.get(token)
            if encoding is None:
                logging.info(f"Token <{token}> has no LAD representation, checking for symantically similar alternatives...")
                encoding = self.handle_unknowns(token)
            vectors.append(encoding)
        vectors = np.asarray(vectors)
        return np.where(np.isnan(vectors), np.nanmean(vectors, axis=0), vectors)


    def handle_unknowns(self, token: str) -> np.ndarray:
        """
        Assumes the original token is not present in the VAD database

        Returns NaN if:
            - original token does not exist in GLoVe
            - original token nor any of the k-nearest words exist in VAD
        :param token:
        :return:
        """
        encoding = self.glove_lexicon.get(token)
        # Token does not exist in VAD, but does in GLoVe
        if encoding is not None:
            return self.get_k_nearest(encoding)
        logging.info(f"Token <{token}> has no GLoVe representation. Representation will be computed from all other tokens in the clause...")
        return [np.nan, np.nan, np.nan]

    def get_k_nearest(self, encoding):
        """

        :param encoding:
        :return: most similar vad-vector within the closest k, if applicable
        """
        # Find 5 most symantically-similar words to encoding
        dist, ndxs = self.glove_searchspace.get('tree').query(encoding.reshape(1, -1), k=self.k)
        for i in ndxs[0]:
            token = self.glove_searchspace.get('labels')[i]
            encoding = self.vad_lexicon.get(token)
            if encoding is not None:
                return encoding
        logging.info(f"Token <{token}> has no LAD-represented words within the {self.k} symantically-closest words.")
        return [np.nan, np.nan, np.nan]


class Tokenizer(object):
    """Preprocess data and handle edge-cases present in the training/test corpus.

        Cleanup process does NOT include token/word standardization."""
    def __init__(self, tokenizer, stopword_language):
        self.tokenizer = tokenizer
        self.stopwords = set(get_stopwords(stopword_language)+nltk.corpus.stopwords.words(stopword_language))

    def __call__(self, clause: str, substitute: bool = True, stopword_removal: bool = True) -> List[str]:
        """

        :param clause:
        :return: words separated
        """
        clause = self.pseudoword_substitution(clause.lower(), substitute=substitute)
        clause = self.space_formatting(clause)
        clause = self.remove_punctuation(clause)
        tokens = self.tokenizer(clause)
        tokens = self.lemmatize(tokens)
        if stopword_removal:
            tokens = self.remove_stop_words(tokens)
        return tokens

    def pseudoword_substitution(self, clause: str, substitute: bool = True) -> str:
        """

        :param clause:
        :return:
        """
        """Basic clause preprocessing convert particular types of text into pseudo words

                :param sentence: sequence of words to clean up
                :return: cleaned sequence of words
                """
        # Convert hyperlinks into pseudo word
        clause = re.sub(r"http[s]?://(?:\w+[.|/]?)+", '<hyp>' if substitute else ' ', clause)
        # Convert decimals or floating points to pseudo word
        clause = re.sub(r"\d*[.]\d+", '<dec>' if substitute else ' ', clause)
        # Convert time's associated with : to pseudo word
        clause = re.sub(r"\d*[.]\d+", '<time>' if substitute else ' ', clause)
        # Convert emails into pseudo word
        clause = re.sub(r"\s(?:\w+[.]?)+@(?:\w+[.]?)", ' <email>' if substitute else ' ', clause)
        # Convert @ mentions into pseudo word
        clause = re.sub(r"\s+@(?:\w+[.]?)+", ' <mention>' if substitute else ' ', clause)
        # Convert all other numerical-only words to pseudo word
        #    e.g. "234-34" to "<num>-<num>", "3PM" to "<num>PM", "234" to "<num>", "53,234" to "<num>"
        clause = re.sub(r"\b\d+(,\d+)*[\s+|$]*", '<num>' if substitute else ' ', clause)
        return clause

    def space_formatting(self, clause: str) -> str:
        # substitute newline character and "p" tags with space
        clause = re.sub(r'<[/]*p>|\n', ' ', clause)
        # Separate num. from letters (e.g. "2PM" to "2 PM", "<num>PM" to "<num> PM")
        clause = re.sub(r'(\d|<num>|<dec>)(\D)', r'\1 \2', clause)
        # Separate non \' punctuation that is not associated with a pseudo word from the surrounding text
        #   (e.g. mid-1990 to "mid 1990" or "It's a wonderful day!" "It's a wonderful day ! ")
        clause = re.sub(
            r"(?<!<hyp)(?<!<dec)(?<!<email)(?<!<num)(?<!<time)(?<!<mention)([^\w\s\d\'])(?!hyp>|dec>|email>|num>|time>|mention>)",
            r' \1 ', clause)
        return clause

    def remove_pseudowords(self, clause: str) -> str:
        return re.sub('<\w+>', '', clause)

    def remove_punctuation(self, clause: str) -> str:
        """Basic clause preprocessing to remove punctuation"""
        # Remove non \' punctuation that is not associated with a pseudo word
        clause = re.sub(
            r"(?<!<hyp)(?<!<dec)(?<!<email)(?<!<num)(?<!<time)(?<!<mention)[^\w\s\d\'](?!hyp>|dec>|email>|num>|time>|mention>)",
            '', clause)
        # Remove non conjunctive \'
        clause = re.sub(r"(?<!\w)\'|\'\s", '', clause)
        return clause

    def remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stopwords]

    def lemmatize(self, tokens):
        # TODO: DO LEMMATIZATION
        return tokens

    def negation_identifier(self, clause):
        """They applied a tagging technique, on unigrams only,

            which flagged all words between the negation word and first punctuation following said word with “NOT_.”
        """
        p = re.compile(r"(?<=\snot\s)(.*?)(?=[^\w\s\'<>]|$)", flags=re.IGNORECASE)
        p = re.compile(r"\s+not\s+(.*?)([^\w\s\'<>]|$)", flags=re.IGNORECASE)
        p = re.compile(r"(?<=\snot)(\s+\w+)+([^\w\s\'<>]|$)", flags=re.IGNORECASE)


def get_sentences(text: str) -> List[str]:
    """

    :param text:
    :return: clauses associated with the input text
    """
    # Identify sentences via "\n.", "!", "?", ";" in addition to the contextually Identify sentences via "."
    tokenizer = nltk.RegexpTokenizer('(?:[.]{1}\n)|[!?;]', gaps=True)
    sentences = nltk.sent_tokenize(text)
    return sentences
