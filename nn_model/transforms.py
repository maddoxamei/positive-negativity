import torch
import torchtext
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import SentencePieceTokenizer
from torchtext.vocab import build_vocab_from_iterator
from dataset import *

#BERT_tokenizer = torchtext.transforms.BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)


class Simple_Tokenizer(object):
    def __init__(self):
        #Cleaning/Word Removal go here
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    def __call__(self,text:str):
        return(self.tokenizer(text))

class Text_Cleaner(object):
    def __init__(self):
        #Cleaning/Word Removal go here
        self.transform_pipeline = get_tokenizer('basic_english') #Placeholder
    def __call__(self,text:str):
        return(self.transform_pipeline(text))

class Vocab_Transform(object):
    def __init__(self,vocab:dict):
        self.vocabulary_dictionary = vocab
        self.tensor_transform = torchtext.transforms.ToTensor()

    def __call__(self,input_tokens:str):
        print(type(input_tokens))
        if type(input_tokens)=='str':
            input_tokens = Simple_Tokenizer(input_tokens)
        output = [self.vocabulary_dictionary[x] if x in self.vocabulary_dictionary else 0 for x in input_tokens]
        return output

