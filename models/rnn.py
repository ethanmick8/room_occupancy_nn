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
            hidden_size=self.hparams.config["hidden_size"],
            num_layers=self.hparams.config["num_layers"],
            batch_first=True
        )
        #self.criterion = nn.functional.mse_loss # loss function - regression
        #self.fc = nn.Linear(self.hparams.model["hidden_size"], 1) # fully connected linear layer - regression
        self.criterion = nn.CrossEntropyLoss() # loss function - classification
        self.fc = nn.Linear(self.hparams.config["hidden_size"], 4) # fully connected linear layer - classification

    def forward(self, x, sequence=None):
        hidden = torch.zeros(self.hparams.config["num_layers"], x.size(0), self.hparams.config["hidden_size"], device=self.device)
        rnn_out, _ = self.rnn(x, hidden) # RNN layer - takes hidden state and input
        if self.hparams.experiment == 0: # MISO
            out = rnn_out[:, -1, :] # (batch_size, hidden_size)
        else: # MIMO
            out = rnn_out # (batch_size, sequence_length, hidden_size)   
        y_pred = self.fc(out) # linear layer    
                
        return y_pred

    # handle training_step and validation_step
    def step_wrapper(self, batch, batch_idx, mode):
        x, y = batch
        #print(f"Input x shape: {x.shape}")  # Expected shape: (batch_size, sequence_length, num_features)
        #print(f"Target y shape: {y.shape}")  # Expected shape: (batch_size, 4)

        y_hat = self(x).view(-1, 4) if self.hparams.experiment == 1 else self(x)
        #print(f"Model output y_hat shape: {y_hat.shape}")  # Expected shape should match y's shape
        #print(f"First 5 predictions: {y_hat[:5]}")
        
        loss = self.criterion(y_hat, y.view(-1)) if self.hparams.experiment == 1 else self.criterion(y_hat, y)
        self.log(mode, loss, batch_size=x.size(0), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'val_loss')
    
    #def test_step(self, batch, batch_idx):
    #    return self.step_wrapper(batch, batch_idx, 'test_loss')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.config["learning_rate"])
        return optimizer