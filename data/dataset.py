from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from ucimlrepo import fetch_ucirepo
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoomOccupancyDataset(Dataset):
    def __init__(self, X, y, batch_size, sequence_length, scale_data=True, mode='MISO'):
        """
        X: features, initially a Pandas DataFrame
        y: targets, initially a Pandas Series or DataFrame
        sequence_length: the length of the input sequences
        scale_data: whether to standardize the data
        mode: Model I/O structure (MISO or MIMO)
        """
        self.mode = mode
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        if scale_data:
            # exclude binary columns from scaling
            binary_cols = [col for col in X.columns if X[col].dropna().value_counts().index.isin([0, 1]).all()]
            binary_data = X[binary_cols]
            numeric_cols = X.drop(columns=binary_cols)
            
            # convert Date and Time columns to useful numeric features
            # day of the week and hour of the day seem most useful
            numeric_cols['Date'] = pd.to_datetime(X['Date'], format='%Y/%m/%d').dt.dayofweek
            numeric_cols['Time'] = pd.to_datetime(X['Time'], format='%H:%M:%S').dt.hour
            
            # scale numeric features
            scaler = StandardScaler()
            numeric_scaled = pd.DataFrame(scaler.fit_transform(numeric_cols), columns=numeric_cols.columns)
            
            # Reset indices to align all columns properly
            numeric_scaled.reset_index(drop=True, inplace=True)
            binary_data.reset_index(drop=True, inplace=True)
            
            # combine scaled features with binary columns
            X_scaled = pd.concat([numeric_cols, binary_data], axis=1)
        else:
            X_scaled = X
            #y = y.to_numpy()

        X_scaled = X_scaled.to_numpy()
        y = y.to_numpy().reshape(-1)
        #print(y.shape)
        
        # Convert class indices to one-hot encoding (not needed)
        #y_one_hot = torch.nn.functional.one_hot(torch.tensor(y), num_classes=4).numpy()
        
        # save the scaler for later use
        scaler_file = 'scaler.pkl'
        joblib.dump(scaler, scaler_file)
        
        self.sequences, self.targets = self.create_sequences(X_scaled, y) 
    
    def create_sequences(self, X, y):
        sequences = []
        targets = []
        for i in range(len(X) - self.sequence_length + 1):
            if self.mode =='SISO':
                seq = np.zeros((self.sequence_length, X.shape[1]))
                seq[0] = X[i]
                sequences.append(seq)
            else: # Multi-input types
                seq = X[i:i + self.sequence_length]
                sequences.append(seq)
            if self.mode == 'MISO' or self.mode == 'SISO':
                # append the target at the end of the sequence
                targets.append(y[i + self.sequence_length - 1])
            else:  # MIMO
                # append targets for each timestep in the sequence
                targets.append(y[i:i + self.sequence_length])
        return np.array(sequences), np.array(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32).to(device), torch.tensor(self.targets[idx], dtype=torch.int64).to(device)