from types import FrameType
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from utils.params import get_params
        
class EJM_LSTM(pl.LightningModule):
    def __init__(self, params):
        super(EJM_LSTM, self).__init__()
        self.save_hyperparameters(params)
        self.lstm = nn.LSTM(
            input_size=self.hparams.data["num_features"],
            hidden_size=self.hparams.model["hidden_size"],
            num_layers=self.hparams.model["num_layers"],
            batch_first=True
        )
        self.criterion = nn.functional.mse_loss # loss function
        # output_size = self.hparams.data["num_features"] -- for predicting all features (not the focus here)
        self.fc = nn.Linear(self.hparams.model["hidden_size"], 1) # fully connected linear layer

    def forward(self, x, sequence=None):
        hidden = torch.zeros(self.hparams.model["num_layers"], x.size(0), self.hparams.model["hidden_size"], device=self.device)
        cell = torch.zeros(self.hparams.model["num_layers"], x.size(0), self.hparams.model["hidden_size"], device=self.device)
        
        lstm_out, _ = self.lstm(x, (hidden, cell))
        
        if self.hparams.experiment == 0: # MISO
            out = lstm_out[:, -1, :] # (batch_size, hidden_size)
        else: # MIMO
            out = lstm_out # (batch_size, sequence_length, hidden_size)
            
        y_pred = self.fc(out) # linear layer
        
        '''# Adjust depending on the experiment type (MISO, MIMO)
        if self.hparams.experiment == 0:
            # MISO
            features, _ = self.lstm(x, (hidden, cell)) # LSTM layer
            out = features[:, -1].view(x.size(0), -1) # using only the last output
            y_pred = self.fc(out) # linear layer
        else:
            # MIMO
            if sequence is None:
                sequence = x.size(1)  # Use the input sequence length if not provided
            y_pred = torch.zeros(x.size(0), sequence, 1).to(x.device)
            for i in range(sequence): # output for every time step
                features, _ = self.lstm(x, (hidden, cell)) # LSTM layer
                out = features[:, -1] # output
                y_pred[:, i, 0] = self.fc(out).squeeze() # linear layer'''
            
        return y_pred

    # handle training_step and validation_step
    def step_wrapper(self, batch, batch_idx, mode):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log(mode, loss, batch_size=x.size(0), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'train_loss')
        
    def validation_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'val_loss')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.model["learning_rate"])
        return optimizer
    
