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
from sklearn.model_selection import ParameterGrid
import datetime

from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from data.dataset import RoomOccupancyDataset
from data.module import RoomOccupancyDataModule
from utils.params import get_params

# currently working on clarifying y, y_hat dimensions and difference for MISO and MIMO within models

def train_model(model_type, params):
    # reproducibility
    seed_everything(42)

    if model_type == 'rnn':
        model = EJM_RNN(params)
    elif model_type == 'lstm':
        model = EJM_LSTM(params)
    else:
        raise ValueError('Model type must be either "rnn" or "lstm"')

    # init data module
    data_module = RoomOccupancyDataModule(batch_size=params['config']['batch_size'],
                                          sequence_length=params['data']['num_sequence'])

    # setup logger and callbacks
    logger = TensorBoardLogger("lightning_logs", name=model_type, version=params['experiment_name'])
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        auto_insert_metric_name=False)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        verbose=True,
        mode='min')

    # init trainer
    trainer = Trainer(
        max_epochs=params["config"]['max_epochs'],
        accelerator='cuda' if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=50)

    # train the model
    trainer.fit(model, datamodule=data_module)

    # returning the best model checkpoint
    return checkpoint_callback.best_model_path

if __name__ == '__main__':
    #torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='Model to train: rnn or lstm')
    parser.add_argument('--grid_search', type=str, default='False', help='Perform grid search: True or False')
    args = parser.parse_args()

    if args.grid_search == 'False':
        params = get_params() # default
        
        params['experiment_name'] = f"default_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        best_model_path = train_model(args.model_type, params)
        print(f"Best model saved at {best_model_path}")
    elif args.grid_search == 'True':
        params = get_params(method='grid_search')
        grid = ParameterGrid(params['config'])

        for param_set in grid:
            params['config'] = param_set
            params['experiment_name'] = f"grid_search/hs:{param_set['hidden_size']}_lr:{param_set['learning_rate']}_bs:{param_set['batch_size']}_nl:{param_set['num_layers']}"
            best_model_path = train_model(args.model_type, params)
            print(f"Best model saved at {best_model_path}")
    else:
        print("Invalid grid_search argument. Must be either 'True' or 'False'")