from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import torch
from ucimlrepo import fetch_ucirepo
from data.dataset import RoomOccupancyDataset
from utils.params import get_params

params = get_params()

class RoomOccupancyDataModule(LightningDataModule):
    """Module that sets up the data for the Room Occupancy Estimation dataset
    and prepares it for training, validation, and testing. It also handles
    cross-validation if specified. The data is fetched from the UCI repository,
    and params are specified via utils/params.py in line with the rest of the 
    code library"""
    def __init__(self, batch_size=32, sequence_length=params['data']['num_sequence'], num_splits=10, is_cross_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        #self.room_occupancy_estimation = fetch_ucirepo(id=864)
        self.x, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.room_occupancy_estimation = None
        self.kfold = TimeSeriesSplit(n_splits=num_splits) # 10-fold cross validation
        self.is_cross_val = is_cross_val
        self.num_splits = num_splits
        self.current_fold = 0
        self.folds = [] # store train and val dataloaders for each fold
        self.initial_train_indices = [] # train indices for the first fold
        self.validation_indices = [] # val indices for each fold

    def prepare_data(self):
        """fetch data from UCI repository"""
        self.room_occupancy_estimation = fetch_ucirepo(id=864)
        self.x = self.room_occupancy_estimation.data.features
        self.y = self.room_occupancy_estimation.data.targets

    def setup(self, stage=None):
        """The setup method is called once before the training starts. It is used to
        split the data into training and validation sets, and to prepare the data
        for training and validation. It also handles cross-validation if specified.
        Args:
            stage (_type_, optional): _description_. Defaults to None.
        """
        # Fetch data if it's not yet loaded
        if not hasattr(self, 'x') or self.x is None:
            self.room_occupancy_estimation = fetch_ucirepo(id=864)
            self.x = self.room_occupancy_estimation.data.features 
            self.y = self.room_occupancy_estimation.data.targets
        if self.is_cross_val:
            if params["experiment"] == 0:
                mode = 'MISO'
            elif params["experiment"] == 1:
                mode = 'MIMO'
            else:
                mode = 'SISO'
            self.validation_indices = []
            for train_index, val_index in self.kfold.split(self.x):
                # split data into training and validation sets
                X_train, X_val = self.x.iloc[train_index], self.x.iloc[val_index]
                y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
                # save indices
                if len(self.folds) == 0: # for handling prediction on first training fold
                    self.initial_train_indices = train_index
                self.validation_indices.append(val_index)
                # reset indices
                X_train, X_val = [df.reset_index(drop=True) for df in [X_train, X_val]]
                y_train, y_val = [s.reset_index(drop=True) for s in [y_train, y_val]]
                # create datasets
                train_dataset = RoomOccupancyDataset(X_train, y_train, self.batch_size, self.sequence_length, mode=mode)
                val_dataset = RoomOccupancyDataset(X_val, y_val, self.batch_size, self.sequence_length, mode=mode)
                # construct a list of tuples containing train and val dataloaders for each fold
                self.folds.append((DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                                DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)))
        else: # default split    
            n = len(self.x)
            train_size = int(n * 0.75) # 75% train, 25% test
            X_train, X_test = self.x.iloc[:train_size], self.x.iloc[train_size:]
            y_train, y_test = self.y.iloc[:train_size], self.y.iloc[train_size:]
            # reset indices
            X_train, X_test = [df.reset_index(drop=True) for df in [X_train, X_test]]
            y_train, y_test = [s.reset_index(drop=True) for s in [y_train, y_test]]
            # save data
            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def train_dataloader(self, fold_index=None):
        """_summary_ - fetches the training dataloader for the current fold
        or the entire training set if cross-validation is not enabled

        Args:
            fold_index (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.is_cross_val: # cross validation
            return self.folds[self.current_fold][0]
        else:
            # specify mode based on the experiment - I/O type for RNN/LSTM
            if params["experiment"] == 0:
                mode = 'MISO'
            elif params["experiment"] == 1:
                mode = 'MIMO'
            else:
                mode = 'SISO'
            train_dataset = RoomOccupancyDataset(self.X_train, self.y_train, self.batch_size, self.sequence_length, mode=mode)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self, fold_index=None):
        if self.is_cross_val: # cross validation
            #
            return self.folds[self.current_fold][1]
        else:
            # RNN/LSTM I/O type
            if params["experiment"] == 0:
                mode = 'MISO'
            elif params["experiment"] == 1:
                mode = 'MIMO'
            else:
                mode = 'SISO'
            val_dataset = RoomOccupancyDataset(self.X_test, self.y_test, self.batch_size, self.sequence_length, mode=mode)
            return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        """Custom dataset, unorthodox dataloader designed for prediction on training and testing sets
        Implemented here only because this route was followed in the seminal paper for the purpose of
        generating confusion matrices seen in their report"""
        # Entire dataset; for widespread prediction
        if params["experiment"] == 0:
            mode = 'MISO'
        elif params["experiment"] == 1:
            mode = 'MIMO'
        else:
            mode = 'SISO'
        dataset = RoomOccupancyDataset(self.x, self.y, self.batch_size, self.sequence_length, mode=mode)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
   
    def train_data(self, fold_index=None):
        if self.is_cross_val:
            # Directly accessing the dataset instead of the DataLoader
            train_dataset = self.folds[self.current_fold][0].dataset
            return train_dataset.all_data()
        else:
            return self.X_train.to_numpy(), self.y_train.to_numpy()

    def val_data(self, fold_index=None):
        if self.is_cross_val:
            # Directly accessing the dataset instead of the DataLoader
            val_dataset = self.folds[self.current_fold][1].dataset
            return val_dataset.all_data()
        else:
            return self.X_test.to_numpy(), self.y_test.to_numpy()
    
    def get_fold_dataloader(self, fold_index):
        """fetch a specific fold's dataloader tuple"""
        return self.folds[fold_index]
    
    def increment_fold(self):
        """helper for cross-validation; increment the current fold"""
        self.current_fold = (self.current_fold + 1) % self.num_splits
        
    