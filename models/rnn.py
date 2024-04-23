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
            input_size=self.hparams["num_features"],
            hidden_size=self.hparams["hidden_size"],
            num_layers=self.hparams["num_layers"],
            batch_first=True
        )
        self.criterion = nn.functional.mse_loss  # loss function
        #output_size = self.hparams.data["num_features"] -- for predicting all features (not the focus here)
        self.fc = nn.Linear(self.hparams["hidden_size"], 1)  # fully connected linear layer = 1 output: occupancy count

    def forward(self, x, sequence=None):
        hidden = torch.zeros(self.hparams["num_layers"], x.size(0), self.hparams["hidden_size"], device=self.device)
        
        rnn_out, _ = self.rnn(x, hidden)
        
        if self.hparams["experiment"] == 0: # MISO
            out = rnn_out[:, -1, :] # (batch_size, hidden_size)
        else: # MIMO
            out - rnn_out # (batch_size, sequence_length, hidden_size)
            
        y_pred = self.fc(out) # linear layer
        
        
        '''if self.hparams["experiment"] == 0:
            # MISO
            features, _ = self.rnn(x, hidden)  # RNN layer - takes hidden state and input
            out = features[:, -1] # using only the last output
            y_pred = self.fc(out) # linear/fully connected layer
        else:
            # MIMO
            if sequence is None: 
                sequence = x.size(1)  # Use the input sequence length if not provided
            y_pred = torch.zeros(x.size(0), sequence, 1).to(x.device)
            for i in range(sequence):  # output for every time step
                features, hidden = self.rnn(x[:, :i+1, :], hidden)  # RNN layer for each timestep
                out = features[:, -1]  # output for each time step
                y_pred[:, i, 0] = self.fc(out).squeeze() # linear layer'''
                
        return y_pred

    # handle training_step and validation_step
    def step_wrapper(self, batch, batch_idx, mode):
        x, y = batch
        #print(f"Input x shape: {x.shape}")  # Expected shape: (batch_size, sequence_length, num_features)
        #print(f"Target y shape: {y.shape}")  # Expected shape: (batch_size, 1)

        y_hat = self(x)
        #print(f"Model output y_hat shape: {y_hat.shape}")  # Expected shape should match y's shape
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        return optimizer