#Torch
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import SentencePieceTokenizer
from torchtext.vocab import build_vocab_from_iterator

#Standard
import numpy as np

#BERT_tokenizer = torchtext.transforms.BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)

class preprocessor(object):
    def __init__(self,tensor_size:int,
                 vocab:dict,
                 tokenizer=torchtext.data.utils.get_tokenizer('basic_english')
                 ):
        self.size = tensor_size
        self.blank = np.zeros(self.size)
        self.tokenizer = tokenizer
        self.vocabulary_dictionary = vocab

    def tokenizer(self,input_str):
        return(self.tokenizer(input_str))

    def cleaner(self,input_str):
        return(input_str) #Placeholder

    def vocab_transform(self,input_tokens:str):
        if type(input_tokens)=='str':
            input_tokens = Simple_Tokenizer(input_tokens)
        output = [self.vocabulary_dictionary[x] if x in self.vocabulary_dictionary else -1 for x in input_tokens]
        return output

    def text_encoder(self,integer_tokens:str):
        #print(integer_tokens)
        if len(integer_tokens) > self.size:
            integer_tokens = integer_tokens[0:self.size]
        elif len(integer_tokens) < self.size:
            temp = self.blank
            temp[0:len(integer_tokens)]=integer_tokens
            integer_tokens = temp
        return np.array(integer_tokens,dtype='int')

    def preprocess(self,lst_input):
        """
        :param lst_input: A list of strings or tokenized strings.
        :return: Tensor ready to input into model.
        """
        output = [self.text_encoder(self.vocab_transform(x)) for x in lst_input]
        output = torch.Tensor(output)
        return(output)