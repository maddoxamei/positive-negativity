import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
import pickle
import argparse
from tqdm import tqdm

"""
Driver script to create the glove searchspace used for vectorizing words
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--glove_lexicon_file", "-f", type=str, required=True, help="Absolute local path (folder+filename+extension) to the GLoVe Lexicon text file"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Read in GLoVe lexicon data
    with open(args.glove_lexicon_file, 'r', encoding='utf8') as file:
        words = []
        vectors = []
        for line in tqdm(file.readlines(), desc=f"Reading lexicon data from {args.glove_lexicon_file}"):
            word, *vector = line.split()
            words.append(word)
            vectors.append(np.asarray(vector).astype(np.float32))
        vectors = np.stack(vectors)
    n, d = vectors.shape
    print(f"{n} found words of {d} dimensions each")

    # Row-wise normalization to make cosine distance translate to euclidean distance
    print("Normalizing word vectors to allow equivalent cosine-similarity using euclidean distance")
    encodings = normalize(vectors, axis=1, norm='l2')
    print("Constructing a BallTree to allow for fast retrieval of similar vectors")
    tree = BallTree(encodings, metric='l2') # Stored in BallTree for fast retrieval

    # Save serialized dictionary
    print("Serializing and saving results")
    with open(args.glove_lexicon_file.rpartition('.')[0]+'_l2.pickle', 'wb') as file:
        pickle.dump({'labels': np.asarray(words), 'tree': tree}, file)

    print("done.")