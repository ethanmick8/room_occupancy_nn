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
import os
from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from models.lda import EJM_LDA
from models.svm import EJM_SVM
from data.dataset import RoomOccupancyDataset
from data.module import RoomOccupancyDataModule
from utils.params import get_params

def train_model(model_type, params, fold_index=None):
    """Trains a model specified by model_type using the given parameters
       either normally or with cross validation if fold_index is provided
       @param model_type: str - model to train: rnn or lstm
       @param params: dict - hyperparameters for the model
       @param fold_index: int - number of folds for cross validation"""
    # reproducibility
    seed_everything(42)

    # initialize model
    if model_type in ['rnn', 'lstm']:
        if model_type == 'rnn':
            model = EJM_RNN(params)
        elif model_type == 'lstm':
            model = EJM_LSTM(params)
    elif model_type == 'lda':
        model = EJM_LDA()
    elif model_type == 'svm':
        model = EJM_SVM(params['svm']['C'], params['svm']['kernel'])
    else:
        raise ValueError('Model type must be "rnn", "lstm", "lda", or "svm"')

    # init data module
    if fold_index is not None:
        is_cross_val = True
        folds = args.fold_index
        print(f"Training model with {folds}-fold cross validation")
    else:
        is_cross_val = False
        folds = 10 # the default value
    data_module = RoomOccupancyDataModule(batch_size=params['config']['batch_size'],
                                          sequence_length=params['data']['num_sequence'],
                                          num_splits=folds,
                                          is_cross_val=is_cross_val)
    # Training process for LDA and SVM logically differs in terms of setup
    if model_type in ['lda', 'svm']:
        if fold_index is None:
            return ValueError("LDA and SVM models must use cross validation. Please provide a fold index.")
        if fold_index > 0: # increment the fold index for cross validation
            data_module.increment_fold()
        data_module.setup(stage='train')
        X_train, y_train = data_module.train_data()
        model.train(X_train, y_train)
        X_val, y_val = data_module.val_data()
        validation_accuracy = model.evaluate(X_val, y_val)
        print(f"Validation Accuracy: {validation_accuracy}")
        
        # Saving the model
        # create directory for of date if not exists
        if not os.path.exists(f"ml_logs/models/{model_type}/{params['experiment_name']}"):
            os.makedirs(f"ml_logs/models/{model_type}/{params['experiment_name']}")
        model_path = f"ml_logs/models/{model_type}/{params['experiment_name']}/{model_type}_{fold_index if fold_index is not None else 'default'}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Logging model path for later reference
        print(f"Model saved at {model_path}")
        return model_path
    else: # RNN or LSTM
        data_module.setup()
        # setup logger and callbacks
        # versioning for tensorboard library - log to a new directory for each experiment
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

        # specifies when and for what reason we should terminate training
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=params["patience"],
            verbose=True,
            mode='min')

        # init trainer - use GPU if available (GPU was used for my project)
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
            trainer.fit(model, datamodule=data_module)

        # returning the best model checkpoint
        return checkpoint_callback.best_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='Model to train: rnn or lstm')
    parser.add_argument('--grid_search', type=str, default='False', help='Perform grid search: True or False')
    parser.add_argument('--fold_index', type=int, default=None, help='Number of folds for cross validation') # future work
    args = parser.parse_args()

    if args.grid_search == 'False':
        params = get_params() # default
        
        if args.fold_index is not None:
            params['experiment_name'] = f"{args.fold_index}_fold/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
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