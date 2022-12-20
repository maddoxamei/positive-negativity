import os
import argparse

import torch.jit

from transforms import *
from dataset_per_document import *
from utils import *
from rules_model import *
from glob import glob
import yaml

# =======================================================
# ===================== LSTM MODEL ======================
# =======================================================
print(os.getcwd())
evaluation_path = '../data'
nrc_lexicon = '../data/lexicons/NRC-VAD-Lexicon'
glove_searchspace = '../data/lexicons/glove.6B.300d_l2'

dataset = Dataset(evaluation_path, nrc_lexicon, glove_searchspace, valence_only=True)
datamodule = torch.utils.data.DataLoader(dataset,
                                           batch_size=4,
                                           num_workers=1,
                                           collate_fn=collate_function)
model = torch.jit.load('model.torchscript').eval()

sentiment_confusion_matrix = np.zeros((2,2))
sentiment_for_thwarted_confusion_matrix = np.zeros((2,2))
for i, (embedded_sentences, _, label) in enumerate(dataset):
    with torch.no_gr
        result = model(embedded_sentences.unsqueeze(0)).round().int().numpy()
    true_sent = int('positive' in dataset.documents[i])
    true_thwart = int('thwart' in dataset.documents[i])
    print(result)

    sentiment_confusion_matrix[true_sent, int(result[0][0])] += 1
    if true_thwart:
        sentiment_for_thwarted_confusion_matrix[true_sent, int(int(result[0][0]))] += 1


# =======================================================
# ===================== RULES MODEL =====================
# =======================================================
sentiment_confusion_matrix = np.zeros((2,2))
thwarted_confusion_matrix = np.zeros((2,2))
sentiment_for_thwarted_confusion_matrix = np.zeros((2,2))

for iter, doc_string in enumerate(dataset.documents):
    print("Evaluating:", doc_string)
    sentences = get_doc_sentences(doc_string, dataset.text_processor)
    # sentence_vectors = get_sentence_sentiments_from_pretrained(sentences, processor, False)
    embeddings = get_embeddings(sentences, dataset.text_processor)
    sentence_vectors = get_sentence_vectors(embeddings, valence_only=True)
    pred_sent, pred_thwart = get_document_sentiment(sentence_vectors, valence_only=True)
    true_sent = int('positive' in doc_string)
    true_thwart = int('thwart' in doc_string)
    print(pred_sent, pred_thwart)

    sentiment_confusion_matrix[true_sent, int(pred_sent)] += 1
    thwarted_confusion_matrix[true_thwart, int(pred_thwart)] += 1
    if true_thwart:
        sentiment_for_thwarted_confusion_matrix[true_sent, int(pred_sent)] += 1

print(sentiment_confusion_matrix)
print(thwarted_confusion_matrix)
print(sentiment_for_thwarted_confusion_matrix)



