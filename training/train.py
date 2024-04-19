import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from data.dataset import RoomOccupancyDataset
from data.module import RoomOccupancyDataModule
from utils.params import get_params

# currently working on clarifying y, y_hat dimensions and difference for MISO and MIMO within models

def train_model(model_type='rnn'):
    # reproducibility
    seed_everything(42)

    # load parameters and setup model
    params = get_params()
    if model_type == 'rnn':
        model = EJM_RNN(params)
    elif model_type == 'lstm':
        model = EJM_LSTM(params)
    else:
        raise ValueError('Model type must be either "rnn" or "lstm"')

    # init data module
    data_module = RoomOccupancyDataModule(batch_size=params["model"]['batch_size'],
                                          sequence_length=params["data"]['num_sequence'])

    # setup logger and callbacks
    logger = TensorBoardLogger("lightning_logs", name=model_type)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        auto_insert_metric_name=False)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min')

    # init trainer
    trainer = Trainer(
        max_epochs=params["model"]['num_epochs'],
        accelerator='cuda' if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=50)

    # train the model
    trainer.fit(model, datamodule=data_module)

    # returning the best model checkpoint
    return checkpoint_callback.best_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='Model to train: rnn or lstm')
    args = parser.parse_args()

    best_model_path = train_model(model_type=args.model_type)
    print(f"Best model saved at {best_model_path}")

# old code
'''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grab the data split into training, validation, and testing (we only care about first 2 here)
X_train, _, y_train, _ = fetch_and_split_data()

# some hyperparameters
sequence_length = 25
batch_size = 32

#print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Val size: {len(X_val)}, Sequence length: {sequence_length}")

# create the datasets
train_dataset = RoomOccupancyDataset(X_train, y_train, sequence_length)
#val_dataset = RoomOccupancyDataset(X_val, y_val, sequence_length)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# last dimension is input size
input_size = train_loader.dataset.sequences.shape[2]

def train_rnn(train_loader, hidden_size=64, num_layers=2, num_epochs=100, learning_rate=0.001):
    model = EJM_RNN(input_size, hidden_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in tqdm(range(num_epochs), desc='Training'):
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
    model = EJM_LSTM(input_size, hidden_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in tqdm(range(num_epochs), desc='Training'):
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
    
    # save the model
    torch.save(trained_model.state_dict(), f'checkpoints/trained_model_{args.model_type}.pkl')'''