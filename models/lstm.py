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
            hidden_size=self.hparams.config["hidden_size"],
            num_layers=self.hparams.config["num_layers"],
            batch_first=True
        )
        self.hidden = None
        self.cell = None
        #self.criterion = nn.functional.mse_loss # loss function - regression
        #self.fc = nn.Linear(self.hparams.config["hidden_size"], 1) # fully connected linear layer - regression
        self.criterion = nn.CrossEntropyLoss() # loss function - classification
        self.fc = nn.Linear(self.hparams.config["hidden_size"], 4) # fully connected linear layer - classification

    def forward(self, x, sequence=None):
        # implement a stateful LSTM, maintaining hidden and cell states across batches
        if self.hidden is None or self.cell is None:
            self.hidden = torch.zeros(self.hparams.config["num_layers"], x.size(0), self.hparams.config["hidden_size"], device=self.device)
            self.cell = torch.zeros(self.hparams.config["num_layers"], x.size(0), self.hparams.config["hidden_size"], device=self.device)
        else:
            self.hidden = self.hidden.detach()
            self.cell = self.cell.detach()
            # resize (applicable for last batch usually)
            self.hidden = self._resize_state(self.hidden, x.size(0))
            self.cell = self._resize_state(self.cell, x.size(0))
        # ensure contigous because of the detach
        hidden = self.hidden.contiguous()
        cell = self.cell.contiguous()
        lstm_out, (self.hidden, self.cell) = self.lstm(x, (hidden, cell))
        if self.hparams.experiment == 0: # MISO
            out = lstm_out[:, -1, :] # (batch_size, hidden_size)
        else: # MIMO
            out = lstm_out # (batch_size, sequence_length, hidden_size)
        y_pred = self.fc(out) # linear layer
        return y_pred

    # handle training_step and validation_step
    def step_wrapper(self, batch, batch_idx, mode):
        x, y = batch
        y_hat = self(x).view(-1, 4) if self.hparams.experiment == 1 else self(x)
        loss = self.criterion(y_hat, y.view(-1)) if self.hparams.experiment == 1 else self.criterion(y_hat, y)
        self.log(mode, loss, batch_size=x.size(0), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'train_loss')
        
    def validation_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'val_loss')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.config["learning_rate"])
        return optimizer
    
    # reset hidden and cell states at the start of each epoch
    '''def on_epoch_start(self):
        self.hidden = None
        self.cell = None'''
        
    '''def on_epoch_end(self):
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.contiguous()
        if self.cell_state is not None:
            self.cell_state = self.cell_state.contiguous()'''
        
    def _resize_state(self, state, batch_size):
        """ Resize the state's batch size while maintaining the state's other dimensions. """
        _, state_batch_size, hidden_size = state.size()
        if state_batch_size != batch_size:
            if batch_size > state_batch_size:
                # Increase the batch size by concatenating zeros
                padding = torch.zeros(self.hparams.config["num_layers"], batch_size - state_batch_size, hidden_size, device=self.device)
                state = torch.cat([state, padding], dim=1)
            else:
                # Decrease the batch size by slicing
                state = state[:, :batch_size, :]
        return state
    
