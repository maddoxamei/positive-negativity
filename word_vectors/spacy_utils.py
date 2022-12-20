from typing import List, Set, Dict, Union

import numpy as np
import sklearn.metrics.pairwise
import spacy
import re

additional_sentence_boundaries: List[str] = [';']
negation_words_phrases: Union[List[str], Set[str]] = ["not"]
negation_words_next: Union[List[str], Set[str]] = ["no", "neither", "nor"]

@spacy.Language.component('sentence_punctuation')
def additional_sentence_parsing(doc):
    for i, token in enumerate(doc):
        if token.lower_ in additional_sentence_boundaries:
            doc[i+1].sent_start = True
            doc[i+1].is_sent_start = True
    return doc

@spacy.Language.component('clausing')
def clause_parsing(doc):
    if len(doc) <= 3:
        return doc
    for i, token in enumerate(doc[:-2]):
        prior_tag = doc[max(i-2, 0)].tag_ if doc[max(i-1, 0)].is_punct else doc[max(i-1, 0)].tag_
        if token.tag_ in ('CC', 'IN') and prior_tag != doc[i+1].tag_:
            # split the document at the following punctuation (due to tokenization process, this excludes apostrophes)
            if doc[max(i - 1, 0)].is_sent_start or doc[max(i - 1, 0)].is_punct:
                for j in range(i, len(doc)-1):
                    if doc[j].is_punct:
                        doc[j+1].is_sent_start = True
                        break
            # split the document at the conjunction if occurring in the middle of the sentence
            if not (doc[max(i-1, 0)].is_sent_start or doc[max(i-1, 0)].is_punct):
                doc[i+1].is_sent_start = True
    return doc

@spacy.Language.component('negation')
def token_negation(doc):
    """ Flags all words between the negation word(s) and first punctuation following said word as negation(s).

    Handles double negative by applying flag and then removing it once second negation occurs.

    :param doc:
    :param negation_words:
    :return:
    """
    for i, token in enumerate(doc):
        if token.lemma_.lower() in negation_words_next:
            doc[i+1]._.set('is_negated', not token._.get('is_negated'))
        if token.lemma_.lower() in negation_words_phrases:
            for j in range(i+1, len(doc)):
                if doc[j].is_punct or doc[j].is_sent_start:
                    break
                doc[j]._.set('is_negated', not token._.get('is_negated'))
    return doc

@spacy.Language.component('pseudo_words')
def identify_pseudowords(doc):
    for token in doc:
        if token.is_digit:
            token.lemma_ = '-DIGIT-'
        elif token.like_email:
            token.lemma_ = '-EMAIL-'
        elif token.like_num:
            token.lemma_ = '-NUMBER-'
        elif token.like_url:
            token.lemma_ = '-URL-'
        elif re.search(r"[\s\b]?@(?:\w+[.]?)+", token.lemma_):
            token.lemma_ = '-MENTION-'

        if re.search(r'-.+-', token.lemma_) is not None:
            token._.set('is_pseudo', True)
    return doc


def customize_stopwords(nlp, additional_stopwords: Union[List[str], Set[str]] = [], remove_stopwords: Union[List[str], Set[str]] = []):
    nlp.Defaults.stop_words |= set(additional_stopwords)
    nlp.Defaults.stop_words -= set(remove_stopwords)

@spacy.Language.factory('substitute_tokenizer', default_config={"infix_re": None, "prefix_re": None, "suffix_re": None})
def substitute_tokenizer(nlp, name, infix_re: List["re.compile"] = None,
                         prefix_re: List["re.compile"] = None,
                         suffix_re: List["re.compile"] = None):
    infix_re = infix_re or nlp.Defaults.infixes
    prefix_re = prefix_re or nlp.Defaults.prefixes
    suffix_re = suffix_re or nlp.Defaults.suffixes
    return spacy.tokenizer.Tokenizer(nlp.vocab, prefix_search=spacy.util.compile_prefix_regex(prefix_re).search,
              suffix_search=spacy.util.compile_suffix_regex(suffix_re).search,
              infix_finditer=spacy.util.compile_infix_regex(infix_re).search,
              token_match=None)

@spacy.Language.factory('additional_tokenizer', default_config={"additional_infix_re": None, "additional_prefix_re": None, "additional_suffix_re": None})
def additional_tokenizer(nlp, name, additional_infix_re: List[str] = None,
                         additional_prefix_re: List["re.compile"] = None,
                         additional_suffix_re: List["re.compile"] = None):
    """
    [r"...", r"..."]

    :param additional_infix_re:
    :param additional_prefix_re:
    :param additional_suffix_re:
    :return:
    """
    infix_re = (additional_infix_re or []) + nlp.Defaults.infixes
    prefix_re = (additional_prefix_re or []) + nlp.Defaults.prefixes
    suffix_re = (additional_suffix_re or []) + nlp.Defaults.suffixes
    return spacy.tokenizer.Tokenizer(nlp.vocab, prefix_search=spacy.util.compile_prefix_regex(prefix_re).search,
              suffix_search=spacy.util.compile_suffix_regex(suffix_re).search,
              infix_finditer=spacy.util.compile_infix_regex(infix_re).search,
              token_match=None)