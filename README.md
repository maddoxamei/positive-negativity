<!-- GETTING STARTED -->
## Getting Started

To get this project running on your local machine:

0. Clone git repo from https://github.com/maddoxamei/positive_negativity
1. Install python (3.8.10)
2. Ensure all libraries from requirements.txt are installed (pip install x)
3. TODO: GloVe pickling
4. Check directories in defaults.yaml (in project directory)
5. Populate data/IMDB_reviews with .txt files.  See Data Set-up.

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
__: Two underscores, related to legacy labelling scheme.
sentimentlabel: Either 'positive' or 'negative'.
thwartinglabel: either 'thwarted' or 'normal'.

Our data was extracted from https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

<!-- USAGE EXAMPLES -->
## Usage
Rules Model
    Run word_vectors/rules_model.py from your python interpreter.
        1. Run results are saved to data/rule_predictions
        2. Confusion matricies are printed.  ***PLEASE NOTE***: 
            True-positive is in bottom-right[1,1], 
            False-positive is in top-right[0,1],
            True-negative in top-left [0,0],
            False-negative is in bottom-left [1,0]
        3. 
            Matrix 1 is sentiment for all documents in data/IMDB_reviews
            Matrix 2 is thwarting flag for all documents in data/IMDB_reviews
            Matrix 3 is sentiment for only thwarted labeled documents in data/IMDB_reviews
        Example:
            [[ 5. 21.]
            [ 1. 23.]]
            [[38.  2.]
            [ 9.  1.]]
            [[2. 2.]
            [0. 6.]]

LSTM
    1. Run word_vectors/main.py.
        model.torchscript saved to word_vectors/ directory
    2. Run evaluation.py
        This uses model.torchscript from previous step
    3. Confusion matricies are printed.  ***PLEASE NOTE***: 
            True-positive is in bottom-right[1,1], 
            False-positive is in top-right[0,1],
            True-negative in top-left [0,0],
            False-negative is in bottom-left [1,0]
    4. 
        Matrix 1 is sentiment for all documents in data/IMDB_reviews
        Matrix 2 is sentiment for only thwarted labeled documents in data/IMDB_reviews
    Example:
        [[33.  0.]
        [ 7.  0.]]
        [[0. 0.]
        [7. 0.]]


<!-- ROADMAP -->
1. Project Directory
    -README: The file you currently have open.
    -requirements.txt: List of necessary python modules for use.
    -defaults.yaml: You can adjust hyperparameters & directory locations here.
    a. data: Data directory
        i. IMDB_reviews: List of text files, one per document.See Data Set-up
        ii. lexicons: Necessary lexicons.
        iii. rule_predictions: Output files for rules model.
    b. word_vectors: Model & Script directory
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
    c. bag_of_words: Contains a bag of words version of the model.  Not tested, possibly not complete.

<!-- CONTACT -->
## Contact

Mei Maddox - [@maddoxamei] - amelia.maddox16@ncf.edu
Thomas FitzGerald - [@thomasfitzgerald87] - thomas.fitzgerald23@ncf.edu

Project Link: [https://github.com/maddoxamei/positive_negativity](https://github.com/maddoxamei/positive_negativity)