<!-- GETTING STARTED -->
## Getting Started

To get this project running on your local machine:

1. Install python (3.8.10)
2. Run `pip install -r requirements.txt` to ensure all necessary libraries are installed
3. Run `py -3 -m spacy download en_core_web_sm` to install the spacy pipeline
4. Download and extract pretrained GLoVe vectors from https://nlp.stanford.edu/data/glove.6B.zip
5. Run `py -3 create_searchspaces.py -f <location_to_extracted_file>` whist located within the `src` directory
6. Download and extract pretrained NRC VAD word vectors from https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip
7. Check `vad_lexicon_file` in src/defaults.yaml and ensure the directories are correct
8. Populate data/IMDB_reviews with .txt files.  See Data Set-up.

### Prerequisites

This model is designed to run on Python version 3.8.10.

### Installation

A full list of required python packages can be found in requirements.txt.

### Data Set-up

By default, your data should be stored in the directory
data\IMDB_reviews
relative to the project directory.

Your data should be stored in text files, in the format

x__sentimentlabel_thwartinglabel.txt

x: ID# of text.
__: Two underscores, related to legacy labeling scheme.
sentimentlabel: Either 'positive' or 'negative'.
thwartinglabel: either 'thwarted' or 'normal'.

The indices for reviews we used are stored in two text files in the project directory, ‘normal_subset’ and ‘thwarted_subset’.  They correspond to the index of the reviews as they appear in the .csv.

Our data was downloaded from: https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

<!-- USAGE EXAMPLES -->

## Evaluating existing models
Navigate to the `src` project folder.

Ensure the src/defaults.yaml file reflects the accurate locations of the `vad_lexicon_file` lexicon text file and `glove_lexicon_file` serialized pickle file created by the script in step 5

To run both the LSTM and Rules-based model on the sample input, run `py -3 evaluation.py`.
To provide custom evaluation data, run `py -3 evaluation.py –evaluation_path <>` with either a directory of reviews formatted in the same way as described above, or a single-file with same restrictions.

This will output two to three confusion matrices based on the model. The upper left corresponds to true-negatives and the bottom right corresponds to true-positives. The rows correspond to ground-truth whereas the columns correspond to the predicted results.

## Retraining LSTM
Navigate to the `src` project folder.

Ensure the src/defaults.yaml file reflects the accurate locations of the `fit_doc_path` training directory, `vad_lexicon_file` lexicon text file, and `glove_lexicon_file` serialized pickle file created by the script in step 5

To retrain the LSTM which uses word vectors, run `py -3 train_word_vectors.py`

Note, this will overwrite the model saved in `artifacts/model.torchscript`.


<!-- ROADMAP -->
1. Project Directory
	-README: The file you currently have open.
	-requirements.txt: List of necessary python modules for use.
	-src/defaults.yaml: You can adjust hyperparameters & directory locations here.
	-normal_subset: indices of non-thwarted reviews
	-thwarted_subset: indices of thwarted reviews
	a. src/data: Data directory
    	i. IMDB_reviews: List of text files, one per document.See Data Set-up
    	ii. lexicons: Necessary lexicons.
    	iii. rule_predictions: Output files for rules model.
	b. src/word_vectors: Model & Script directory
    	i. dataset_per_clause
    	ii. dataset_per_document
    	iii. evaluation
        		evaluation file for LSTM
    	iv. main
        		training file for LSTM
    	v. model
    	vi. rules_model
        		run file for rules-model
    	vii. spacy_utils
    	viii. transforms
        		transforms for dataset pipeline
    	ix. utils
	c. src/bag_of_words: Contains a bag of words version of the model.  In previous iteration, the model successfully trained, however due to reformatting of reviews it is unrunable as-is now.

<!-- CONTACT -->
## Contact

Mei Maddox - [@maddoxamei] - amelia.maddox16@ncf.edu
Thomas FitzGerald - [@thomasfitzgerald87] - thomas.fitzgerald23@ncf.edu