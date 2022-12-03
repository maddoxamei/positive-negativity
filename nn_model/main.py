import torch
import pytorch_lightning as pl

from transforms import *

#Dataset
from dataset import *

def __main__():

    ref_doc_path = r'C:\File\GitHub\Positive-Negativity\data' #Debug
    data = Dataset(ref_doc_path)
    vocab_transformer = Vocab_Transform(data.vocab)