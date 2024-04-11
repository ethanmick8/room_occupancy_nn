import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
from torch.utils.data import DataLoader

from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from datasets.dataset import RoomOccupancyDataset, fetch_and_split_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grab the data split into training, validation, and testing (we only care about first 2 here)
X_train, X_val, _, y_train, y_val, _ = fetch_and_split_data()

# some hyperparameters
sequence_length = 25
batch_size = 32

# create the datasets
train_dataset = RoomOccupancyDataset(X_train, y_train, sequence_length)
val_dataset = RoomOccupancyDataset(X_val, y_val, sequence_length)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# last dimension is input size
input_size = train_loader.dataset.sequences.shape[2]

def train_rnn(train_loader, hidden_size=64, num_layers=2, num_epochs=100, learning_rate=0.001):
    model = EJM_RNN(input_size, hidden_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    return model


def train_lstm(train_loader, hidden_size=64, num_layers=2, num_epochs=100, learning_rate=0.001):
    model = EJM_LSTM(input_size, hidden_size, num_layers)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

if __name__ == '__main__':
    # parse the model type
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='rnn or lstm')
    args = parser.parse_args()
    
    if args.model_type == 'rnn':
        trained_model = train_rnn(train_loader)
    elif args.model_type == 'lstm':
        trained_model = train_lstm(train_loader)
    else:
        raise ValueError('Model type must be rnn or lstm')
    
    # pickle and save the model
    with open(f'../checkpoints/trained_model_{args.model_type}.pkl', 'wb') as f:
        pickle.dump(trained_model, f)