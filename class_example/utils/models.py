"""
Purpose: Model Tools
"""


import torch
import lightning as L

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class Network(L.LightningModule):
    """
    Purpose: LSTM Network
    """
    
    def __init__(self, params):
        """
        Purpose:
        - Define network architecture
        
        Arguments:
        - params (dict[any]): user defined parameters
        """
        
        super().__init__()

        self.alpha = params["model"]["learning_rate"]
        self.num_epochs = params["model"]["num_epochs"]
        
        self.hidden_size = params["model"]["hidden_size"]
        self.num_layers = params["model"]["num_layers"]
        
        self.input_size = params["data"]["num_features"]

        self.task = params["experiment"]

        # Create: LSTM Architecture

        self.arch = torch.nn.LSTM(batch_first=True,
                                  num_layers=self.num_layers,
                                  input_size=self.input_size, 
                                  hidden_size=self.hidden_size)

        self.linear = torch.nn.Linear(self.hidden_size, 1)
     
    def objective(self, labels, preds):
        """
        Purpose:
        - Define network objective function
        
        Arguments:
        - labels (torch.tensor[float]): ground truths
        - preds (torch.tensor[float]): model predictions
        
        Returns:
        - (torch.tensor[float]): error between ground truths and predictions
        """
        
        return torch.nn.functional.mse_loss(labels, preds)

    def configure_optimizers(self):
        """
        Purpose: 
        - Define network optimizers and schedulars

        Returns:
        - (dict[str, any]): network optimizers and schedulars
        """

        # Create: Optimzation Routine

        optimizer = Adam(self.parameters(), lr=self.alpha)

        # Create: Learning Rate Schedular

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def forward(self, x, target_seq=None):
        """
        Purpose: 
        - Define network forward pass

        Arguments:
        - x (torch.tensor[float]): network input observation
        - target_seq (int): number of sequence predictions

        Returns:
        - (torch.tensor[float]): network predictions
        """

        batch_size = x.size()[0]

        # Create: Hidden & Cell States
        
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Task: Many To One
        
        if self.task == 0:
            features, (hidden, cell) = self.arch(x, (hidden, cell))
            features = features[:, -1].view(batch_size, -1)
            preds = self.linear(features)

        # Task: Many To Many
        
        else:
            
            preds = torch.zeros(batch_size, target_seq).to(x.device)

            for i in range(target_seq):
                features, (hidden, cell) = self.arch(x, (hidden, cell))
                features = features[:, -1].view(batch_size, -1)
                output = self.linear(features).view(-1)
                preds[:, i] = output

        return preds

    def shared_step(self, batch, batch_idx, tag):
        """
        Purpose: 
        - Define network batch processing procedure

        Arguments: 
        - batch (tuple[any]): network observations (i.e., samples, labels)
        - batch_idx (int): batch iteration counter
        - tag (str): tag for labeling analytics

        Returns:
        - (torch.tensor[any]): error between ground-truth and predictions
        """
        
        samples, labels = batch
        batch_size = samples.size()[0]

        # Gather: Predictions
        
        if self.task == 0:
            preds = self(samples)
        else:
            target_seq = labels.size()[1]
            preds = self(samples, target_seq)

        # Calculate: Objective
 
        loss = self.objective(preds, labels)

        self.log(tag, loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Purpose: 
        - Define network training iteration

        Arguments: 
        - batch (tuple[any]): network observations (i.e., samples, labels)
        - batch_idx (int): batch iteration counter

        Returns:
        - (Function): batch processing procedure that returns training error
        """

        return self.shared_step(batch, batch_idx, "train_error")

    def validation_step(self, batch, batch_idx):
        """
        Purpose: 
        - Define network validation iteration

        Arguments: 
        - batch (tuple[any]): network observations (i.e., samples, labels)
        - batch_idx (int): batch iteration counter
        """

        self.shared_step(batch, batch_idx, "valid_error")