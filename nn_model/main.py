#Torch

import torch
import torchtext
import torch.nn as nn

from torchtext.data.utils import get_tokenizer
from torchtext.transforms import SentencePieceTokenizer
from torchtext.vocab import build_vocab_from_iterator

import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger

#SkLearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Standard
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from dataset import *
from model import *
from transforms import *

model_parameters = dict(
    embedding_size = 1000,
    hidden_size = 100,
    seq_len = 24,
    batch_size = 512,
    num_layers = 3,
    dropout = 0.2,
    learning_rate = 0.001,
    criterion = nn.CrossEntropyLoss(),
    max_epochs = 1,
    n_features = 7,
    shuffle = True,
    num_workers = 1,
)

other_parameters = dict(
    ref_doc_path = r'C:\File\GitHub\Positive-Negativity\data',
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english'),
)

#def __main__():

print('Retrieving dataset.')
data = Dataset(ref_doc_path = other_parameters['ref_doc_path'],
               tokenizer = other_parameters['tokenizer'],
               debug_flag = False
               )

print('Initializing preprocessor.')
pp = preprocessor(tensor_size=model_parameters['embedding_size'],
                  vocab=data.vocab,
                  tokenizer=other_parameters['tokenizer']
                  )

print('Running DataModule setup.')
data_module = DataModule(data.text_array,
                         data.label_array,
                         pp,
                         model_parameters['seq_len'],
                         model_parameters['batch_size'],
                         model_parameters['num_workers'],
                         model_parameters['shuffle']
                         )

print('Initializing model.')
model = LSTM_Classifier(model_parameters['embedding_size'],
                        model_parameters['hidden_size'],
                        model_parameters['seq_len'],
                        model_parameters['batch_size'],
                        model_parameters['num_layers'],
                        model_parameters['dropout'],
                        model_parameters['learning_rate'],
                        model_parameters['criterion']
                        )

csv_logger = CSVLogger('./', name='lstm', version='0'),

print('Initializing trainer.')
trainer = pl.Trainer(
    max_epochs=model_parameters['max_epochs'],
    logger=csv_logger,
    #gpus=1,
    #row_log_interval=1,
    #progress_bar_refresh_rate=2,
)

print('Fitting trainer.')
trainer.fit(model, datamodule=data_module)
print('Testing.')
trainer.test(model, datamodule=data_module)

