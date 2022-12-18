import torch
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class LSTM_Classifier(pl.LightningModule):
    def __init__(self,
                 n_input_features,
                 hidden_size,
                 num_layers,
                 output_size,
                 dropout,
                 learning_rate,
                 **kwargs):

        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.BCELoss() if output_size == 1 else nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(input_size=n_input_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, text, text_lengths=None):
        """

        :param text: padded batch of variable length encoded text (B T *)
        :param text_lengths: list of text lengths of each batch element
        :return:
        """
        batch_size = text.size(0)
        h_0 = Variable(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size))
        c_0 = Variable(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size))
        # h_0 = Variable(torch.randn(1, batch_size, self.hparams.hidden_size))
        # c_0 = Variable(torch.randn(1, batch_size, self.hparams.hidden_size))

        packed_input = pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input, (h_0, c_0))
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        text_features = self.linear(output[:, -1])
        # if len(text_features.shape) == 3:
        #     text_features = text_features.squeeze(1)

        return torch.sigmoid(text_features)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x, clause_lengths, y = batch
        y_hat = self.forward(x, clause_lengths)
        loss = self.criterion(y_hat.type(torch.float32), y.type(torch.float32))
        # TODO: log los
        return loss

    def validation_step(self, batch, batch_idx):
        x, clause_lengths, y = batch
        y_hat = self.forward(x, clause_lengths)
        loss = self.criterion(y_hat.type(torch.float32), y.type(torch.float32))
        # TODO: log los
        return loss