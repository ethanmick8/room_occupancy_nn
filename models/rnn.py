from types import FrameType
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from utils.params import get_params

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EJM_RNN(pl.LightningModule):
    def __init__(self, params):
        super(EJM_RNN, self).__init__()
        self.save_hyperparameters(params)
        self.rnn = nn.RNN(
            input_size=self.hparams.data["num_features"],
            hidden_size=self.hparams.model["hidden_size"],
            num_layers=self.hparams.model["num_layers"],
            batch_first=True
        )
        self.criterion = nn.functional.mse_loss  # loss function
        # Output size always total number of features; reshaped later depending on experiment type
        output_size = self.hparams.data["num_features"]
        self.fc = nn.Linear(self.hparams.model["hidden_size"], output_size)  # fully connected linear layer

    def forward(self, x, sequence=None):
        hidden = torch.zeros(self.hparams.model["num_layers"], x.size(0), self.hparams.model["hidden_size"], device=self.device)
        # Adjust based on type
        if self.hparams.experiment == 0:
            # MISO
            features, _ = self.rnn(x, hidden)  # RNN layer - hidden state
            out = features[:, -1].view(x.size()[0], -1)  # using only the last output
            y_pred = self.fc(out)  # linear layer
            #features = features[:, -1] # using only the last output
        else:
            # MIMO
            y_pred = torch.zeros(x.size()[0], sequence).to(x.device)
            for i in range(sequence):  # output for every time step
                features, hidden = self.rnn(x, hidden)  # RNN layer
                out = features[:, i].view(x.size()[0], -1)  # output for each time step
                out = self.fc(out).view(-1)  # linear layer
                y_pred[:, i] = out

        return y_pred

    # handle training_step and validation_step
    def step_wrapper(self, batch, batch_idx, mode):
        x, y = batch
        
        print(f"Input x shape: {x.shape}")  # Expected shape: (batch_size, sequence_length, num_features)
        print(f"Target y shape: {y.shape}")  # Expected shape: (batch_size, sequence_length, num_features)
        
        if self.hparams.experiment == 1: # MIMO
            # 3D - batch size, sequence length, number of features
            y = y.view(x.size(0), -1, self.hparams.data["num_features"])
            print(f"MIMO Target y reshaped shape: {y.shape}")  # Expected shape: (batch_size, sequence_length, num_features)

        y_hat = self(x, y.size(1)) if self.hparams.experiment == 1 else self(x)
        print(f"Model output y_hat shape: {y_hat.shape}")  # Expected shape should match y's shape
        loss = self.criterion(y_hat, y)
        self.log(mode, loss, batch_size=x.size(0), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'val_loss')
    
    #def test_step(self, batch, batch_idx):
    #    return self.step_wrapper(batch, batch_idx, 'test_loss')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.model["learning_rate"])
        return optimizer