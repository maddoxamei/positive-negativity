import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

#Need:
#   text,
#   labels,
#   vocabulary

class LSTM_Classifier(pl.LightningModule):
    def __init__(self,embedding_length,hidden_size,output_size=1):
        super().__init__()
        self.embedding_length = embedding_length
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.embedding_length,self.hidden_size)
        self.label = nn.Linear(self.hidden_size,self.output_size)

    def forward(self):
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))

        return self.label(final_hidden_state[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #self being the model?
        loss = torch.nn.CrossEntropyLoss(y_hat,y) #Prediction, actual?
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)