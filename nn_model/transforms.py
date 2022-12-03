import torch
import torchtext
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import SentencePieceTokenizer
from torchtext.vocab import build_vocab_from_iterator

basic_tokenizer = get_tokenizer('basic_english')
torchtext.data.utils.get_tokenizer('basic_english')
#BERT_tokenizer = torchtext.transforms.BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)


strTest = 'This is text.'

class TextTransform(object):
    def __init__(self, language):
        self.tokenizer = torchtext.data.utils.get_tokenizer(language)

        self.tensor_transform = torchtext.transforms.ToTensor()

    def __call__(self,text:str):

        return(self.tokenizer(text))

print(strTest)
transform = TextTransform('basic_english')

print(transform(strTest))