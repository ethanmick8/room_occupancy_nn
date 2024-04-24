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

def train_model(model_type, params, fold_index=None):
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
                                          sequence_length=params['data']['num_sequence'],
                                          is_cross_val=True if fold_index else False)

    # setup logger and callbacks
    if fold_index is not None:
        version = f"{params['experiment_name']}/fold_{fold_index}"
    else:
        version = params['experiment_name']
    logger = TensorBoardLogger("lightning_logs", name=model_type, version=version)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        auto_insert_metric_name=False)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
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
    if not fold_index: # default split
        trainer.fit(model, datamodule=data_module)
    else: # cross validation
        trainer.fit(model, data_module.train_dataloader(fold_index), data_module.val_dataloader(fold_index))

    # returning the best model checkpoint
    return checkpoint_callback.best_model_path

if __name__ == '__main__':
    #torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='Model to train: rnn or lstm')
    parser.add_argument('--grid_search', type=str, default='False', help='Perform grid search: True or False')
    parser.add_argument('--fold_index', type=int, default=None, help='Number of folds for cross validation') # future work
    args = parser.parse_args()

    if args.grid_search == 'False':
        params = get_params() # default
        
        if args.fold_index == "True":
            params['experiment_name'] = f"cross_val/default_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            for i in range(args.fold_index): # k-fold CV
                best_model_path = train_model(args.model_type, params, fold_index=i)
                print(f"Best model for fold {i} saved at {best_model_path}")
        else:
            params['experiment_name'] = f"default_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            best_model_path = train_model(args.model_type, params)
            print(f"Best model saved at {best_model_path}")
    elif args.grid_search == 'True':
        params = get_params(method='grid_search')
        grid = ParameterGrid(params['config'])

        for param_set in grid:
            params['config'] = param_set
            params['experiment_name'] = f"grid_search/hs:{param_set['hidden_size']}_lr:{param_set['learning_rate']}_bs:{param_set['batch_size']}_nl:{param_set['num_layers']}"
            if args.fold_index:
                for i in range(args.fold_index): # k-fold CV
                    best_model_path = train_model(args.model_type, params, fold_index=i)
                    print(f"Best model for fold {i} saved at {best_model_path}")
            else:
                best_model_path = train_model(args.model_type, params)
                print(f"Best model saved at {best_model_path}")
    else:
        print("Invalid grid_search argument. Must be either 'True' or 'False'")