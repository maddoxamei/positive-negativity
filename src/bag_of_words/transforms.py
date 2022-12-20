import re
from typing import List, Union, Hashable, Set, NoReturn

import nltk
import torchtext
from stopwords import get_stopwords
import numpy as np

class BagOfWordsEncoder():
    """
    Functions similarly to the sklearn.preprocessing.LabelEncoder, however it handles encoding words which are not present in the training corpa
    """
    def __init__(self, unknown_token: Hashable = '<unknown>') -> NoReturn:
        """
        :param unknown_token: token for unknown words which may be encountered during testing
        """
        self.unknown_token = unknown_token
        self.unknown_position = 0

        self.classes_ = np.array([self.unknown_token])
        self.n_classes = len(self.classes_)
        self.class_to_int = dict(zip(self.classes_, range(len(self.classes_))))

    def fit(self, y: Union[str, Set[str], List[str]]) -> NoReturn:
        vocabulary = y if isinstance(y, set) else set
        vocabulary.add(self.unknown_token)

        self.classes_ = np.asarray(sorted(vocabulary))
        self.class_to_int = dict(zip(self.classes_, range(len(self.classes_))))

        self.unknown_position = self.class_to_int.get(self.unknown_token)

    def transform(self, y: Union[str, List[str]]) -> List[int]:
        return [self.class_to_int.get(class_, self.unknown_position) for class_ in y]


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
            r' \1 ', "Although I like cheese, eggs, and bacon, I'd prefer veggies the most")
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
        p=re.compile(r"(?<=\snot\s)(.*?)(?=[^\w\s\'<>]|$)", flags=re.IGNORECASE)
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
