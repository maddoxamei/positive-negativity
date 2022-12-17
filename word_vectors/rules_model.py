import pandas as pd
import os
import torch
import math

from word_vectors.transforms import *
from glob import glob
encoder = WordVectorEncoder('data/lexicons/NRC-VAD-Lexicon.txt', 'data/lexicons/glove.6B.50d.txt')
tokenizer = Tokenizer(torchtext.data.utils.get_tokenizer('basic_english'), 'english')
documents = list(glob('data/IMDB_reviews/*.txt'))

def get_doc_sentences(path):
    with open(path, 'r') as file:
        return get_sentences(file.read())

def get_embeddings(sentences):
    """

    :param document_index:
    :return: torch.Size([49, 8, 3])
    """
    embeddings = []
    for sent in sentences:
        tokens = tokenizer(sent, False, True)
        encoding = torch.as_tensor(encoder(tokens))
        #print(len(tokens), encoding.shape)
        embeddings.append(encoding)
    return embeddings

def sentence_extremity(embeddings):
    """

    :param embeddings:
    :return: A dataframe of VAD values.  One word is returned per sentence, determined by which word's valence was furthest from .5.
    """
    df = []
    for i in embeddings:
        nump = i.numpy()
        temp_nump = abs(nump-.5)
        most_extreme_valence = max(temp_nump[:,0])
        temp_index = np.argmax(temp_nump[:,0]==most_extreme_valence)
        ext_val = nump[temp_index]
        df.append(ext_val)
    return(pd.DataFrame(df))

def thwarting_predictor(embeddings):
    """
    Predicts if a text is thwarting or not based on valence of the first 4/5th of the text VS the last 1/5.
    Additional weight is given to the last sentence for long texts.
    :param embeddings:
    :return: Boolean, True if thwarting.
    """
    df = sentence_extremity(embeddings)
    row_count = len(df)
    if row_count<5:
        return False
    overall_mean_valence = np.mean(df.iloc[:,0].array)
    if overall_mean_valence==.5:
        return False
    break_point = math.floor(row_count*.8)
    early = df.iloc[:break_point,0]
    late = df.iloc[break_point:,0]
    main_mean_valence = np.mean(early.array)

    if len(late)>2:
        #Increases weight of last sentence in longer documents
        #print(late)
        late = np.append(late,np.repeat(late.array[-1],math.floor(len(late)/2)))
        #print(late,np.mean(late.array))
        ending_mean_valence = np.mean(late)
    else:
        ending_mean_valence = np.mean(late.array)
    #print(main_mean_valence,ending_mean_valence)
    if overall_mean_valence>.5:
        #Positive
        if (ending_mean_valence <= main_mean_valence - .2) | (ending_mean_valence <= main_mean_valence/2):
            return True
    elif overall_mean_valence<.5:
        #Negative
        if (ending_mean_valence >= main_mean_valence + .2) | (ending_mean_valence >= main_mean_valence*2):
            return True
    else:
        return False

def sentence_average(embeddings):
    df = []
    for i in embeddings:
        df.append(i.mean(axis=0).numpy())
    return(pd.DataFrame(df))

def every_word(embeddings):
    df = []
    for i in embeddings:
        for j in i.numpy():
            df.append(abs(j-.5))


documents = list(glob('data/IMDB_reviews/*.txt'))

strDir = r'C:\File\GitHub\positive-negativity'
for iter,doc_string in enumerate(documents):
    print(doc_string)
    sentences = get_doc_sentences(os.path.join(strDir,doc_string))
    embeddings = get_embeddings(sentences)
    result = thwarting_predictor(embeddings)
    print(result)

    #df = sentence_average(embeddings)
    #df.to_csv(r'C:\File\Active_Dataset\\NLP_'+doc_string[18:20]+'.csv')