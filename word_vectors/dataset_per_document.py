from typing import Optional, Any, List, Tuple
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder
import pytorch_lightning as pl
import os
import numpy as np
from transforms import *
from utils import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 fit_doc_path: str,
                 vad_lexicon_file: str,
                 glove_lexicon_file: str,
                 valence_only,
                 **kwargs
                 ):
        self.documents = [
            os.path.join(fit_doc_path, path)
            for path in os.listdir(fit_doc_path)
            if path.endswith((".txt"))
        ]
        self.text_processor = TextProcessor(vad_lexicon_file+'.txt', glove_lexicon_file+'.pickle')
        self.label_encoder = self.setup()
        self.valence_only = valence_only

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int):
        """

        :param clause_idx: desired clause index, NOT document index
        :return:
        """
        sentences = get_doc_sentences(self.documents[idx], self.text_processor)
        # sentence_vectors = get_sentence_sentiments_from_pretrained(sentences, self.text_processor)
        embeddings = get_embeddings(sentences, self.text_processor)
        sentence_vectors = get_sentence_vectors(embeddings, valence_only=self.valence_only) # np.ndarray of shape (NUM_OF_SENTENCES, )
        sentence_vectors = np.expand_dims(sentence_vectors, 1) # np.ndarray of shape (NUM_OF_SENTENCES, )

        label = os.path.basename(self.documents[idx]).rsplit('_', maxsplit=2)[1]
        label = self.label_encoder.transform([[label]])
        # label = [['positive' in self.documents[idx]]]

        return torch.as_tensor(sentence_vectors).type(torch.float32), len(sentence_vectors), torch.as_tensor(label).type(torch.float32)

    def setup(self):
        labels = set([])
        for doc in self.documents:
            labels.add(os.path.basename(doc).rsplit('_', maxsplit=2)[1])

        label_encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
        label_encoder.fit(np.asarray(sorted(labels)).reshape((-1, 1)))

        return label_encoder

def collate_function(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, List[int]]:
    """

    :param batch: e.g. [(tensor([4, 7, 2]), 3), (tensor([16,  9,  5, 14]), 4)]
    :return: e.g.
    """
    documents, lengths, labels = zip(*batch)
    padded_documents = pad_sequence(documents, batch_first=True)
    return padded_documents, lengths, torch.stack(labels).squeeze(1)


class DataModule(pl.LightningDataModule):
    def __init__(self, fit_doc_path: str, batch_size: int = 128, num_workers: int = 1, train_split: float = 0.8, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.save_hyperparameters()
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = Dataset(self.hparams.fit_doc_path, **self.kwargs)
        split = int(self.hparams.train_split * len(dataset))
        self.train_set, self.val_set = torch.utils.data.random_split(
            dataset,
            [split, len(dataset) - split],
        )

    def train_dataloader(self) -> "TRAIN_DATALOADERS":
        return torch.utils.data.DataLoader(self.train_set,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.num_workers,
                                           collate_fn=collate_function)

    def val_dataloader(self) -> "EVAL_DATALOADERS":
        return torch.utils.data.DataLoader(self.val_set,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.num_workers,
                                           collate_fn=collate_function)
