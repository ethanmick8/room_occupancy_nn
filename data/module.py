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
    def __init__(self, batch_size=32, sequence_length=params['data']['num_sequence'], num_splits=10, is_cross_val=False):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        #self.room_occupancy_estimation = fetch_ucirepo(id=864)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.room_occupancy_estimation = None
        self.kfold = TimeSeriesSplit(n_splits=10) # 10-fold cross validation
        self.is_cross_val = is_cross_val
        self.num_splits = num_splits
        self.current_fold = 0
        self.folds = [] # store train and val dataloaders for each fold

    def prepare_data(self):
        # fetch data from UCI repository
        self.room_occupancy_estimation = fetch_ucirepo(id=864)
        self.x = self.room_occupancy_estimation.data.features
        self.y = self.room_occupancy_estimation.data.targets

    def setup(self, stage=None):
        # Fetch data if it's not yet loaded
        if not hasattr(self, 'x') or self.x is None:
            self.room_occupancy_estimation = fetch_ucirepo(id=864)
            self.x = self.room_occupancy_estimation.data.features 
            self.y = self.room_occupancy_estimation.data.targets
        if self.is_cross_val: # (incomplete)
            if params["experiment"] == 0:
                mode = 'MISO'
            elif params["experiment"] == 1:
                mode = 'MIMO'
            else:
                mode = 'SISO'
            for train_index, val_index in self.kfold.split(self.x):
                X_train, X_val = self.x.iloc[train_index], self.x.iloc[val_index]
                y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
                print(X_train.shape, X_val.shape)
                print(y_train.shape, y_val.shape)
                train_dataset = RoomOccupancyDataset(X_train, y_train, self.batch_size, self.sequence_length, mode=mode)
                val_dataset = RoomOccupancyDataset(X_val, y_val, self.batch_size, self.sequence_length, mode=mode)
                self.folds.append((DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                                DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)))
        else: # default split    
            n = len(self.x)
            train_size = int(n * 0.75) # 75% train, 25% test
            X_train, X_test = self.x.iloc[:train_size], self.x.iloc[train_size:]
            y_train, y_test = self.y.iloc[:train_size], self.y.iloc[train_size:]
            
            X_train, X_test = [df.reset_index(drop=True) for df in [X_train, X_test]]
            y_train, y_test = [s.reset_index(drop=True) for s in [y_train, y_test]]
            
            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def train_dataloader(self, fold_index=None):
        if fold_index: # cross validation
            return self.folds[self.current_fold][0]
        else:
            if params["experiment"] == 0:
                mode = 'MISO'
            elif params["experiment"] == 1:
                mode = 'MIMO'
            else:
                mode = 'SISO'
            train_dataset = RoomOccupancyDataset(self.X_train, self.y_train, self.batch_size, self.sequence_length, mode=mode)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self, fold_index=None):
        if fold_index: # cross validation
            return self.folds[self.current_fold][1]
        else:
            if params["experiment"] == 0:
                mode = 'MISO'
            elif params["experiment"] == 1:
                mode = 'MIMO'
            else:
                mode = 'SISO'
            val_dataset = RoomOccupancyDataset(self.X_test, self.y_test, self.batch_size, self.sequence_length, mode=mode)
            return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        # Entire dataset; for widespread prediction
        if params["experiment"] == 0:
            mode = 'MISO'
        elif params["experiment"] == 1:
            mode = 'MIMO'
        else:
            mode = 'SISO'
        dataset = RoomOccupancyDataset(self.x, self.y, self.batch_size, self.sequence_length, mode=mode)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def increment_fold(self):
        self.current_fold = (self.current_fold + 1) % self.num_splits