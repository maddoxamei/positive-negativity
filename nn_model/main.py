#Torch

import torch
import torchtext
import torch.nn as nn

from torchtext.data.utils import get_tokenizer
from torchtext.transforms import SentencePieceTokenizer
from torchtext.vocab import build_vocab_from_iterator

import pytorch_lightning as pl

#SkLearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Standard
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from dataset import *
from transforms import *
from model import *

model_parameters = dict(
    seq_len = 24,
    batch_size = 512,
    criterion = nn.CrossEntropyLoss(),
    max_epochs = 10,
    n_features = 7,
    hidden_size = 100,
    num_layers = 3,
    dropout = 0.2,
    learning_rate = 0.001,
)

def __main__():

    ref_doc_path = r'C:\File\GitHub\Positive-Negativity\data' #Debug
    data = Dataset(ref_doc_path)

    test = [['people', 'will', 'do', 'great'], ['first', 'its', 'them']]
    Vocab_Transform = Vocab_Transform(data.vocab)
    Text_Encoder = Text_Encoder(1000)
    preprocessor = preprocessor(Vocab_Transform, Text_Encoder)
    model = LSTM_Classifier(model_parameters)

