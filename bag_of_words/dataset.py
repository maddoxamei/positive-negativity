from typing import Optional, Any, List, Tuple
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder
import pytorch_lightning as pl
import os
import numpy as np
from transforms import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 ref_doc_path: str,
                 tokenizer: str,
                 stopword_language: str,
                 **kwargs
                 ):
        self.documents = [
            os.path.join(ref_doc_path, path)
            for path in os.listdir(ref_doc_path)
            if path.endswith((".txt"))
        ]
        self.tokenizer = Tokenizer(torchtext.data.utils.get_tokenizer(tokenizer), stopword_language)
        self.token_encoder = BagOfWordsEncoder()
        self.clause_to_document_index, self.label_encoder = self.setup()

    def __len__(self) -> int:
        return self.clause_to_document_index[-1]

    def __getitem__(self, clause_idx: int):
        """

        :param clause_idx: desired clause index, NOT document index
        :return:
        """
        document_index = np.argmax(self.clause_to_document_index > clause_idx)
        # Handles edge case of obtaining relative indexes from the first document
        n_clauses = self.clause_to_document_index[max(0, document_index-1)]
        # relative clause index within the document that contains it
        relative_idx = clause_idx - n_clauses

        with open(self.documents[document_index], 'r') as file:
            clause = get_sentences(file.read())[relative_idx]
        tokens = self.tokenizer(clause)

        # TODO: THIS LINE WAS CHANGED (was maxsplit=2)[1][relative_idx]
        label = os.path.basename(self.documents[document_index]).split('_', maxsplit=3)[2]
        # TODO: DO NOT CHANGE THIS LINE
        label = self.label_encoder.transform([[label]])

        return torch.as_tensor(self.token_encoder.transform(tokens)), len(tokens), torch.as_tensor(label)

    def setup(self):
        vocabulary = set([])  # set ensures duplicates are not created when adding new elements
        labels = set([])
        """
        The LSTM classifies clauses. Therefore we want a dataset of clauses, NOT a dataset of documents.
        However since a single document can have multiple clauses, we need to keep track of which document contains which subset of clauses.
        We track this by keeping an ordered cumulative sum of clauses in new_index[]
        """
        clause_to_document_index = [0]

        #Iterate through each document, tokenizing as a set of clauses.
        for doc in self.documents:
        # TODO: THIS LINE WAS CHANGED (was maxsplit=2)[1][relative_idx]
            labels.update(list(os.path.basename(doc).split('_', maxsplit=3)[2]))
            with open(doc, 'r') as file:
                clauses = get_sentences(file.read())
            # Associate clause indexes to a particular document index
            new_index = clause_to_document_index[-1] + len(clauses)
            clause_to_document_index.append(new_index)
            # Update known vocabulary words/tokens
            for clause in clauses:
                tokens = self.tokenizer(clause)
                vocabulary.update(tokens)

        #Use vocabulary set to create a reference dictionary. These will be used to translate data for the model.
        self.token_encoder.fit(vocabulary)

        label_encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
        label_encoder.fit(np.asarray(sorted(labels)).reshape((-1, 1)))

        return np.asarray(clause_to_document_index[1:]), label_encoder


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

    def collate_function(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, List[int]]:
        """

        :param batch: e.g. [(tensor([4, 7, 2]), 3), (tensor([16,  9,  5, 14]), 4)]
        :return: e.g.
        """
        # Separate clauses and their respective lengths into separate variables
        #   clauses: [tensor([4, 7, 2]), tensor([16,  9,  5, 14])]
        #   lengths: [3, 4]
        clauses, lengths, labels = zip(*batch)
        # Pad
        #   tensor([[ 4,  7,  2,  0],
        #           [16,  9,  5, 14]]
        padded_clauses = pad_sequence(clauses, batch_first=True)
        return padded_clauses, lengths, torch.stack(labels).squeeze(1)

    def train_dataloader(self) -> "TRAIN_DATALOADERS":
        return torch.utils.data.DataLoader(self.train_set,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.num_workers,
                                           collate_fn=self.collate_function)

    def val_dataloader(self) -> "EVAL_DATALOADERS":
        return torch.utils.data.DataLoader(self.val_set,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=self.hparams.num_workers,
                                           collate_fn=self.collate_function)
