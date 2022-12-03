import os
import numpy as np
import psutil
from typing import Optional, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from transforms import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ref_doc_path: str, transformation):
        self.documents = [
            os.path.join(ref_doc_path, path)
            for path in os.listdir(ref_doc_path)
            if path.endswith((".csv"))
        ]

        self.transform = transform

        self.text_array = np.empty(0)
        self.label_array = np.empty(0)
        self.setup()

        self.vocab = self.build_vocabulary(self.text_array)
        self.label_encoder = LabelEncoder()
        #self.label_encoder.fit(self.reference_labels)

    def __len__(self) -> int:
        return len(self.text_array)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return (self.text_array[idx],self.label_array[idx])

    def setup(self):
        for doc in self.documents:
            self.text_array = np.append(self.text_array,np.loadtxt(doc,delimiter=',',quotechar='"',dtype=str,encoding='utf8',usecols=(0)))
                #Label column may need changing
            self.label_array = np.append(self.label_array,np.loadtxt(doc,delimiter=',',quotechar='"',dtype=str,encoding='utf8',usecols=(1)))
        #Train/Val/Test split
        return(self.text_array,self.label_array)

    def train_validation_test_split(self):
        print('not done yet')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    def build_vocabulary(self):
        text_array = self #Does this do anything?
        flat_list = [item for subl in text_array for item in basic_tokenizer(subl)]
        flat_list = np.unique(flat_list)
        #print(flat_list)
        int_list = np.array(range(len(flat_list)))+1
        #print(int_list)
        output_dict = dict(zip(flat_list, int_list))
        #print(output_dict)
        return(output_dict)

    def text_to_tensor(self,input_text):
        #Input must be tokenized
        return(self.vocab[x] for x in input_text)



