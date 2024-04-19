from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

from data.dataset import RoomOccupancyDataset

class RoomOccupancyDataModule(LightningDataModule):
    def __init__(self, batch_size=32, sequence_length=30):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.room_occupancy_estimation = fetch_ucirepo(id=864)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def prepare_data(self):
        # Fetch data here
        self.room_occupancy_estimation = fetch_ucirepo(id=864)

    def setup(self, stage=None):
        # Split data
        X = self.room_occupancy_estimation.data.features 
        y = self.room_occupancy_estimation.data.targets
        
        n = len(X)
        train_size = int(n * 0.75) # 75% train, 25% test
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        X_train, X_test = [df.reset_index(drop=True) for df in [X_train, X_test]]
        y_train, y_test = [s.reset_index(drop=True) for s in [y_train, y_test]]
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def train_dataloader(self):
        train_dataset = RoomOccupancyDataset(self.X_train, self.y_train, self.sequence_length)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = RoomOccupancyDataset(self.X_test, self.y_test, self.sequence_length)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)