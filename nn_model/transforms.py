import torch
import torchtext
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import SentencePieceTokenizer
from torchtext.vocab import build_vocab_from_iterator
from dataset import *

#BERT_tokenizer = torchtext.transforms.BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)

class text_encoder(object):
    def __init__(self,tensor_size:int):
        self.size = tensor_size
        self.blank = np.zeros(self.size)
        self.tensor_transform = torchtext.transforms.ToTensor()

    def __call__(self,integer_tokens:str):
        if len(integer_tokens) > self.size:
            integer_tokens = integer_tokens[0:self.size]
        elif len(integer_tokens) < self.size:
            temp = self.blank
            temp[0:len(integer_tokens)]=integer_tokens
            integer_tokens = temp
        return(self.tensor_transform(integer_tokens))

class Simple_Tokenizer(object):
    def __init__(self):
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    def __call__(self,text:str):
        return(self.tokenizer(text))

class Text_Cleaner(object):
    def __init__(self):
        #Cleaning/Word Removal go here
        self.cleaner = get_tokenizer('basic_english') #Placeholder
    def __call__(self,text:str):
        return(self.cleaner(text))

class Vocab_Transform(object):
    def __init__(self,vocab:dict):
        self.vocabulary_dictionary = vocab
        self.tensor_transform = torchtext.transforms.ToTensor()

    def __call__(self,input_tokens:str):
        print(type(input_tokens))
        if type(input_tokens)=='str':
            input_tokens = Simple_Tokenizer(input_tokens)
        output = [self.vocabulary_dictionary[x] if x in self.vocabulary_dictionary else 0 for x in input_tokens]
        output = self.tensor_transform(output)
        return output

