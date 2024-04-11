from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoomOccupancyDataset(Dataset):
    def __init__(self, X, y, sequence_length, scale_data=True):
        """
        X: features, initially a Pandas DataFrame
        y: targets, initially a Pandas Series or DataFrame
        sequence_length: the length of the input sequences
        scale_data: whether to standardize the data
        """
        if scale_data:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            y = scaler.fit_transform(y.values.reshape(-1, 1))
            
        self.sequences, self.targets = self.create_sequences(X, y, sequence_length) 
    
    def create_sequences(self, X, y, sequence_length):
        sequences = []
        targets = []
        for i in range(len(X) - sequence_length):
            sequences.append(X[i:i+sequence_length])
            targets.append(y[i+sequence_length])
        return np.array(sequences), np.array(targets)
       
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32).to(device), torch.tensor(self.targets[idx], dtype=torch.float32).to(device)

def fetch_and_split_data():
    # fetch dataset from UCI
    room_occupancy_estimation = fetch_ucirepo(id=864)
    # data (as pandas dataframes)
    X = room_occupancy_estimation.data.features 
    y = room_occupancy_estimation.data.targets
    # train/val/test split of 70/15/15
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test