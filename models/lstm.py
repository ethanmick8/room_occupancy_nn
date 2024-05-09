from types import FrameType
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from utils.params import get_params
        
class EJM_LSTM(pl.LightningModule):
    """_summary_ This class is a PyTorch Lightning Module that defines the LSTM model
    for the Room Occupancy dataset. It is used to serve as the model for training,
    validation, and testing. The model consists of an LSTM layer, a linear layer, and
    a loss function. The model is used in the training script to train the model and
    evaluate its performance.

    Args:
        pl (_type_): PyTorch Lightning Module
    """
    def __init__(self, params):
        """ Initialize the LSTM model with the given hyperparameters.
        Args:
            params (_type_): _description_ The hyperparameters for the LSTM model"""
        super(EJM_LSTM, self).__init__()
        self.save_hyperparameters(params)
        # define the LSTM layer with the given hyperparameters
        self.lstm = nn.LSTM(
            input_size=self.hparams.data["num_features"],
            hidden_size=self.hparams.config["hidden_size"],
            num_layers=self.hparams.config["num_layers"],
            batch_first=True
        )
        self.hidden = None # default init
        self.cell = None # default init
        self.criterion = nn.CrossEntropyLoss() # loss function - classification
        # fully connected linear layer - classification
        self.fc = nn.Linear(self.hparams.config["hidden_size"], 4) 
    def forward(self, x, sequence=None):
        # reset for each sequence
        self.hidden = torch.zeros(self.hparams.config["num_layers"], 
                                  x.size(0), self.hparams.config["hidden_size"], 
                                  device=self.device)
        self.cell = torch.zeros(self.hparams.config["num_layers"], 
                                x.size(0), self.hparams.config["hidden_size"], 
                                device=self.device)
        # (alt) implement a stateful LSTM, maintaining hidden and cell states 
        # across batches (used in the paper , commented out when not needed)
        # This is useful for training on sequences that are longer than the 
        # batch size (see stateful section for additional context)
        '''if self.hidden is None or self.cell is None:
            self.hidden = torch.zeros(self.hparams.config["num_layers"], 
            x.size(0), self.hparams.config["hidden_size"], device=self.device)
            self.cell = torch.zeros(self.hparams.config["num_layers"], 
            x.size(0), self.hparams.config["hidden_size"], device=self.device)
        else:
            self.hidden = self.hidden.detach()
            self.cell = self.cell.detach()
            # resize (applicable for last batch usually)
            self.hidden = self._resize_state(self.hidden, x.size(0))
            self.cell = self._resize_state(self.cell, x.size(0))
        # ensure contigous because of the detach
        hidden = self.hidden.contiguous()
        cell = self.cell.contiguous()'''
        # LSTM layer - takes hidden and cell states and input
        lstm_out, _ = self.lstm(x, (self.hidden, self.cell)) 
        if self.hparams.experiment == 0 or self.hparams.experiment == 2: # MISO/SISO
            out = lstm_out[:, -1, :] # (batch_size, hidden_size)
        else: # MIMO
            out = lstm_out # (batch_size, sequence_length, hidden_size)
        y_pred = self.fc(out) # linear layer
        return y_pred

    def step_wrapper(self, batch, batch_idx, mode):
        """_summary_ Wrapper function for the training and validation steps

        Args:
            batch (_type_): _description_ The batch of data
            batch_idx (_type_): _description_ The index of the batch
            mode (_type_): _description_ The mode of the step - train or validation

        Returns:
            _type_: _description_ The loss for the step
        """
        x, y = batch
        # y_hat - y from the forward pass specific computation of loss detailed in the paper
        y_hat = self(x).view(-1, 4) if self.hparams.experiment == 1 else self(x)
        loss = self.criterion(y_hat, y.view(-1)) if self.hparams.experiment == 1 else self.criterion(y_hat, y)
        self.log(mode, loss, batch_size=x.size(0), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'train_loss')
        
    def validation_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'val_loss')

    def configure_optimizers(self):
        """Configure the optimizer for the LSTM model. Adam is used in all cases and 
        the learning rate is specified in the hyperparameters."""
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.config["learning_rate"])
        return optimizer
        
    def _resize_state(self, state, batch_size):
        """ Resize the state's batch size while maintaining the state's other dimensions. This
        is useful for maintaining the hidden and cell states across batches when training on
        sequences that are longer than the batch size. (Stateful LSTM training)"""
        _, state_batch_size, hidden_size = state.size()
        if state_batch_size != batch_size:
            if batch_size > state_batch_size:
                # Increase the batch size by concatenating zeros
                padding = torch.zeros(self.hparams.config["num_layers"], 
                                      batch_size - state_batch_size, 
                                      hidden_size, device=self.device)
                state = torch.cat([state, padding], dim=1)
            else:
                # Decrease the batch size by slicing
                state = state[:, :batch_size, :]
        return state
    
