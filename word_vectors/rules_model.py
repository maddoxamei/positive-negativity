import os

from transforms import *
from utils import *
from glob import glob
import yaml


def get_sections(sentence_vectors):
    four_fifths = int(len(sentence_vectors) * 4 / 5)
    return sentence_vectors[:four_fifths], sentence_vectors[four_fifths:]

def get_section_sentiment(section_average, valence_only):
    """
    :param section_average:
    :return: True if postive, False if negative
    """
    # When looked at valence when determining sentiment
    if valence_only:
        return bool(section_average > .5)
    # When looked at both valence and arousal when determining sentiment
    else:
        return bool(section_average > .5*.2)


def get_document_sentiment(sentence_vectors, valence_only, threshold=.2):
    first_section, second_section = get_sections(sentence_vectors)

    # Entire document sentiment is simply the "first section" sentiment
    # if there is only one sentence in the entire document
    if len(second_section) == 0:
        return get_section_sentiment(first_section, valence_only=valence_only)

    # Get section averages
    first_section_average = np.mean(first_section)
    second_section_average = np.mean(second_section)

    # Increases weight of last sentence in longer documents
    if len(second_section) > 2:
        second_section_average = np.append(second_section, np.repeat(second_section[-1], len(second_section) // 2)).mean()

    first_section_sentiment = get_section_sentiment(first_section_average, valence_only=valence_only)
    second_section_sentiment = get_section_sentiment(second_section_average, valence_only=valence_only)

    # If the first and second sections have different sentiments, and
    #   the difference between the sentiments is sufficient, then
    #   Indicate there is the presence of thwarting.
    #   Otherwise, indicate the lack of thwarting
    different_sentiments = first_section_sentiment != second_section_sentiment
    thwarted = different_sentiments and np.abs(first_section_average-second_section_average) > threshold

    # If there is no thwarting, entire document sentiment is simply the "first section" sentiment
    # If there is IS thwarting, entire document sentiment is simply the "second section" sentiment
    return second_section_sentiment if thwarted else first_section_sentiment


documents = list(glob('../data/IMDB_reviews/*.txt'))
processor = TextProcessor('../data/lexicons/NRC-VAD-Lexicon.txt', '../data/lexicons/glove.6B.50d.pickle')
strDir = r'..'

with open(os.path.join(strDir, 'defaults.yaml'), 'r') as file:
    defaults = yaml.safe_load(file)

for iter,doc_string in enumerate(documents):
    print(doc_string)
    sentences = get_doc_sentences(doc_string, processor)
    sentence_vectors = get_sentence_sentiments_from_pretrained(sentences, processor, False)
    # embeddings = get_embeddings(sentences, processor)
    # sentence_vectors = get_sentence_vectors(embeddings, valence_only=defaults.get('valence_only'))
    result = get_document_sentiment(sentence_vectors, valence_only=defaults.get('valence_only'))
    # not sure what you were wanting to print and why
    np.savetxt(os.path.join(strDir, "data", "rule_predictions",f"{os.path.basename(doc_string).rpartition('.')[0]}_{result}_sentence_metrics.csv"), sentence_vectors)