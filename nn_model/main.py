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
    max_epochs = 10,
    n_features = 7,
    shuffle = True
)

other_parameters = dict(
    ref_doc_path = r'C:\File\GitHub\Positive-Negativity\data',
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english'),
)

def __main__():

    data = Dataset(other_parameters['ref_doc_path'],
                   other_parameters['tokenizer']
                   )

    pp = preprocessor(model_parameters['embedding_size'],
                      data.vocab,
                      other_parameters['tokenizer']
                      )

    data_module = DataModule(pp.preprocess(),
                             model_parameters['seq_len'],
                             model_parameters['batch_size'],
                             model_parameters['num_workers'],
                             model_parameters['shuffle']
                             )

    model = LSTM_Classifier(model_parameters['embedding_size'],
                            model_parameters['hidden_size'],
                            model_parameters['seq_len'],
                            model_parameters['batch_size'],
                            model_parameters['num_layers'],
                            model_parameters['dropout'],
                            model_parameters['learning_rate'],
                            model_parameters['criterion']
                            )

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=model_parameters['max_epochs'],
        accumulate_grad_batches=model_parameters['batch_size'] // datamodule.hparams.batch_size,
        callbacks=[
            Logging(datamodule.val_set.reference_images[0], args.predefined_model),
            TransferLearning(args.max_epochs // 2),
            KAdjustment(args.max_epochs // 2),
            AccuracyMeasurement(args.predefined_model, every_n_epochs=None, supertype_card_reference=supertype_card_reference)
        ],
        fast_dev_run=args.debug,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False)
    )

    trainer.fit(model, datamodule=data_module)