from types import FrameType
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from utils.params import get_params

class EJM_RNN(pl.LightningModule):
    """_summary_ This class is a PyTorch Lightning Module that defines the RNN model 
    for the Room Occupancy dataset. It is used to serve as the model for training,
    validation, and testing. The model consists of an RNN layer, a linear layer, and
    a loss function. The model is used in the training script to train the model and
    evaluate its performance.

    Args:
        pl (_type_): PyTorch Lightning Module
    """
    def __init__(self, params):
        super(EJM_RNN, self).__init__()
        self.save_hyperparameters(params)
        self.rnn = nn.RNN(
            input_size=self.hparams.data["num_features"],
            hidden_size=self.hparams.config["hidden_size"],
            num_layers=self.hparams.config["num_layers"],
            batch_first=True
        )
        self.hidden = None
        self.criterion = nn.CrossEntropyLoss() # loss function - classification
        # fully connected linear layer - classification
        self.fc = nn.Linear(self.hparams.config["hidden_size"], 4) 

    def forward(self, x, sequence=None):
        # non-stateful
        self.hidden = torch.zeros(self.hparams.config["num_layers"], 
                                  x.size(0), self.hparams.config["hidden_size"], 
                                  device=self.device)
        # (alt) stateful approach - see LSTM model for more details
        '''if self.hidden is None:
            self.hidden = torch.zeros(self.hparams.config["num_layers"], 
            x.size(0), self.hparams.config["hidden_size"], device=self.device)
        else:
            self.hidden = self.hidden.detach()
            self.hidden = self._resize_state(self.hidden, x.size(0))
        hidden = self.hidden.contiguous()'''
        rnn_out, _ = self.rnn(x, self.hidden) # RNN layer - hidden state and input
        if self.hparams.experiment == 0 or self.hparams.experiment == 2: # MISO/SISO
            out = rnn_out[:, -1, :] # (batch_size, hidden_size)
        else: # MIMO
            out = rnn_out # (batch_size, sequence_length, hidden_size)   
        y_pred = self.fc(out) # linear layer    
                
        return y_pred

    # handle training_step and validation_step
    def step_wrapper(self, batch, batch_idx, mode):
        x, y = batch
        # loss computation - cross entropy loss - y_hat is the predicted output
        # y_hat - y_pred for timestep t
        y_hat = self(x).view(-1, 4) if self.hparams.experiment == 1 else self(x)    
        loss = self.criterion(y_hat, y.view(-1)) if self.hparams.experiment == 1 else self.criterion(y_hat, y)
        self.log(mode, loss, batch_size=x.size(0), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.step_wrapper(batch, batch_idx, 'val_loss')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.config["learning_rate"])
        return optimizer
    
    def _resize_state(self, state, batch_size):
        """ Resize the state's batch size while maintaining the state's other dimensions. """
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