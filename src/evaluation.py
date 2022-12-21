import argparse
from src.word_vectors.dataset_per_document import *
from src.word_vectors.rules_model import *
from tqdm import tqdm
import yaml

"""
Driver script to create the glove searchspace used for vectorizing words
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--evaluation_path", "-eval", type=str, default="../artifacts/example__positive_thwarted.txt", help="Local path to the file (folder+filename+extension) or file-containing directory (folder) for model evaluation purposes"
)

if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(0)

    with open('defaults.yaml', 'r') as file:
        defaults = yaml.safe_load(file).get('datamodule')

    dataset = Dataset(args.evaluation_path, defaults.get('vad_lexicon_file'), defaults.get('glove_lexicon_file'), valence_only=defaults.get('valence_only'))
    datamodule = torch.utils.data.DataLoader(dataset,
                                             batch_size=4,
                                             num_workers=1,
                                             collate_fn=collate_function)

    # =======================================================
    # ===================== LSTM MODEL ======================
    # ======================================================='

    model = torch.jit.load('../artifacts/model.torchscript').eval()

    sentiment_confusion_matrix = np.zeros((2,2))
    sentiment_for_thwarted_confusion_matrix = np.zeros((2,2))
    for i, (embedded_sentences, _, label) in tqdm(enumerate(dataset), desc="Evaluating the LSTM model"):
        with torch.no_grad():
            result = model(embedded_sentences.unsqueeze(0)).round().int().numpy()
        true_sent = int('positive' in dataset.documents[i])
        true_thwart = int('thwart' in dataset.documents[i])

        sentiment_confusion_matrix[true_sent, int(result[0][0])] += 1
        if true_thwart:
            sentiment_for_thwarted_confusion_matrix[true_sent, int(int(result[0][0]))] += 1

    print("====================================")
    print("======== LSTM MODEL RESULTS ========")
    print("====================================")
    print("Confusion matrix for predicting overall sentiment on ALL evaluation documents")
    print(sentiment_confusion_matrix)
    print("Confusion matrix for predicting overall sentiment on ONLY thwarted evaluation documents")
    print(sentiment_for_thwarted_confusion_matrix)

    # =======================================================
    # ===================== RULES MODEL =====================
    # =======================================================
    sentiment_confusion_matrix = np.zeros((2,2))
    thwarted_confusion_matrix = np.zeros((2,2))
    sentiment_for_thwarted_confusion_matrix = np.zeros((2,2))

    for iter, doc_string in tqdm(enumerate(dataset.documents), desc="Evaluating the Rules-Based model"):
        sentences = get_doc_sentences(doc_string, dataset.text_processor)
        # sentence_vectors = get_sentence_sentiments_from_pretrained(sentences, processor, False)
        embeddings = get_embeddings(sentences, dataset.text_processor)
        sentence_vectors = get_sentence_vectors(embeddings, valence_only=True)
        pred_sent, pred_thwart = get_document_sentiment(sentence_vectors, valence_only=True)
        true_sent = int('positive' in doc_string)
        true_thwart = int('thwart' in doc_string)

        sentiment_confusion_matrix[true_sent, int(pred_sent)] += 1
        thwarted_confusion_matrix[true_thwart, int(pred_thwart)] += 1
        if true_thwart:
            sentiment_for_thwarted_confusion_matrix[true_sent, int(pred_sent)] += 1

    print("====================================")
    print("===== RULE-BASED MODEL RESULTS =====")
    print("====================================")
    print("Confusion matrix for predicting overall sentiment on ALL evaluation documents")
    print(sentiment_confusion_matrix)
    print("Confusion matrix for predicting overall sentiment on ONLY thwarted evaluation documents")
    print(sentiment_for_thwarted_confusion_matrix)
    print("Confusion matrix for predicting the existance of thwarting on ALL evaluation documents")
    print(thwarted_confusion_matrix)



