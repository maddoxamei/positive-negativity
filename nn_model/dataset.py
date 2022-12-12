import os
import numpy as np
from typing import Optional, Dict, List, Tuple

import pytorch_lightning as pl
import torch

#sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torchtext
import pandas as pd

from main import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 ref_doc_path: str,
                 tokenizer=torchtext.data.utils.get_tokenizer('basic_english')
                 ):

        self.documents = [
            os.path.join(ref_doc_path, path)
            for path in os.listdir(ref_doc_path)
            if path.endswith((".csv"))
        ]

        #self.tokenizer = Simple_Tokenizer
        self.tokenizer = tokenizer

        # self.df = pd.DataFrame({'text': pd.Series(dtype='str'),
        #            'label': pd.Series(dtype='str'),
        #            'filename': pd.Series(dtype='str'),
        #            'index': pd.Series(dtype='int')})
        self.text_array = np.empty(0)
        self.label_array = np.empty(0)
        self.vocab = {}

        self.setup()

        self.label_encoder = LabelEncoder()
        #self.label_encoder.fit(self.reference_labels)

    def __len__(self) -> int:
        return len(self.text_array)

    def __getitem__(self, idx: int):
        return (self.text_array[idx],self.label_array[idx])

    def setup(self):
        for doc in self.documents:
            csv_input = pd.read_csv(doc)
            self.text_array = csv_input.iloc[0:,0]
            self.label_array = labels = csv_input.iloc[0:,1]
            # text = csv_input.iloc[0:,0]
            # labels = csv_input.iloc[0:,1]  #Label column may need changing
            # file_array = np.array([os.path.basename(doc)]*len(labels))
            # self.df = self.df.append(pd.DataFrame(text,labels,file_array))

        # self.df.insert(3, 'index', range(len(self.df)))
        # self.text_array = self.df.text
        # self.label_array = self.df.label

        #This part was build_vocabulary, couldn't get it to work as a separate function
        flat_list = [item for subl in self.text_array for item in self.tokenizer(subl)]
        flat_list = pd.value_counts(flat_list)
        flat_list = (flat_list[flat_list > 5]).index.to_numpy()
        int_list = np.array(range(len(flat_list))) + 1

        #if debug: #Not implemented yet
        flat_list = flat_list[0:100]
        self.label_array = self.label_array[0:100]
        int_list = int_list[0:100]

        self.vocab = dict(zip(flat_list, int_list))
        return(self.text_array,self.label_array,self.vocab)

class DataModule(pl.LightningDataModule):
    def __init__(self, text_preprocess, seq_len = 1, batch_size = 128, num_workers=0, shuffle = False):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.stratify = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.columns = None
        self.text_preprocess = text_preprocess

    def setup(self, stage=None):
        # X = self.df[['text','index']]
        # Y = self.df[['label','index']]

        X = self.text_array
        Y = self.label_array

        if self.shuffle:
            self.stratify = True

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=True, stratify=self.stratify
        )

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=0.25, shuffle=True, stratify=self.stratify
        )

        if stage == 'fit' or stage is None:
            self.X_train = self.text_preprocess(X_train)
            self.Y_train = Y_train.values.reshape((-1, 1))
            self.X_val = self.text_preprocess(X_val)
            self.Y_val = Y_val.values.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.X_test = self.text_preprocess(X_test)
            self.Y_test = Y_test.values.reshape((-1, 1))


    def train_dataloader(self):
        train_dataset = Dataset(self.X_train,
                                          self.y_train,
                                          seq_len=self.seq_len)
        train_loader = Dataset(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = Dataset(self.X_val,
                                        self.y_val,
                                        seq_len=self.seq_len)
        val_loader = Dataset(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = Dataset(self.X_test,
                                         self.y_test,
                                         seq_len=self.seq_len)
        test_loader = Dataset(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle,
                                 num_workers=self.num_workers)

        return test_loader