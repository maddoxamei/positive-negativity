import os
import numpy as np
from typing import Optional, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
import torchtext
from transforms import *
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ref_doc_path: str,):
        self.documents = [
            os.path.join(ref_doc_path, path)
            for path in os.listdir(ref_doc_path)
            if path.endswith((".csv"))
        ]

        #self.tokenizer = Simple_Tokenizer
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english') #Should come from transforms, was producing a circular import error.

        self.text_array = np.empty(0)
        self.label_array = np.empty(0)
        self.vocab = {}

        self.setup()

        #self.build_vocabulary()
        self.label_encoder = LabelEncoder()
        #self.label_encoder.fit(self.reference_labels)

    def __len__(self) -> int:
        return len(self.text_array)

    def __getitem__(self, idx: int):
        return (self.text_array[idx],self.label_array[idx])

    def setup(self):
        for doc in self.documents:
            #self.text_array = np.append(self.text_array,np.loadtxt(doc,delimiter=',',quotechar='"',dtype=str,encoding='utf8',usecols=(0)))
            #self.label_array = np.append(self.label_array,np.loadtxt(doc,delimiter=',',quotechar='"',dtype=str,encoding='utf8',usecols=(1)))
            csv_input = pd.read_csv(doc)
            self.text_array = csv_input.iloc[0:,0]
            self.label_array = csv_input.iloc[0:,1]  #Label column may need changing

        #This part was build_vocabulary
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

    def build_vocabulary(self):
        #Working version embedded in setup
        flat_list = np.unique([item for subl in self.text_array for item in self.tokenizer(subl)])
        print(flat_list)
        int_list = np.array(range(len(flat_list)))+1
        print(int_list)
        output_dict = dict(zip(flat_list, int_list))
        print(output_dict)
        return(output_dict)

    def train_validation_test_split(self):
        print('not done yet')





