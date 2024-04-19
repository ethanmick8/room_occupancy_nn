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
        # Output size always total number of features; reshaped later depending on experiment type
        output_size = self.hparams.data["num_features"]
        self.fc = nn.Linear(self.hparams.model["hidden_size"], output_size) # fully connected linear layer

    def forward(self, x, sequence=None):
        hidden = torch.zeros(self.hparams.model["num_layers"], x.size(0), self.hparams.model["hidden_size"], device=self.device)
        cell = torch.zeros(self.hparams.model["num_layers"], x.size(0), self.hparams.model["hidden_size"], device=self.device)
        #out, _ = self.lstm(x, (hidden, cell))
        
        # Adjust depending on the experiment type (MISO, MIMO)
        if self.hparams.experiment == 0:
            # MISO
            features, _ = self.lstm(x, (hidden, cell)) # LSTM layer
            out = features[:, -1].view(x.size()[0], -1) # using only the last output
            y_pred = self.fc(out) # linear layer
        else:
            # MIMO
            y_pred = torch.zeros(x.size()[0], sequence).to(x.device)
            
            for i in range(sequence): # output for every time step
                features, _ = self.lstm(x, (hidden, cell)) # LSTM layer
                out = features[:, -1].view(x.size()[0], -1) # output
                out = self.fc(out).view(-1) # linear layer
                y_pred[:, i] = out
            
        return y_pred

    # handle training_step and validation_step
    def step_wrapper(self, batch, batch_idx, mode):
        x, y = batch
        
        if self.hparams.experiment == 1: # MIMO
            # 3D - batch size, sequence length, number of features
            y = y.view(x.size(0), -1, self.hparams.data["num_features"])
        
        y_hat = self(x, y.size(1)) if self.hparams.experiment == 1 else self(x)
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
    
# old code
'''class EJM_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, task):
        super(EJM_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Assuming output dimension is 1
        self.task = task
        
    def forward(self, x):
        hidden = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(x.device)
        cell = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (hidden, cell))
        out = self.fc(out[:, -1, :]) # Use only the last output
        return out'''