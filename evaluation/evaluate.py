import torch
import numpy as np
import argparse
from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from pytorch_lightning import Trainer
from tqdm import tqdm
import joblib
import os
import yaml
import pandas as pd
from torch.nn.functional import softmax

from data.module import RoomOccupancyDataModule
from data.dataset import RoomOccupancyDataset
from utils.metrics import calculate_metrics, plot_confusion_matrix, plot_occupancy_with_time, display_grid_results
from utils.params import get_params

params = get_params() # default fine for all cases here

def load_model(model_type, checkpoint_path):
    # dynamically load the model from checkpoint
    if model_type == 'rnn':
        model = EJM_RNN.load_from_checkpoint(checkpoint_path)
    elif model_type == 'lstm':
        model = EJM_LSTM.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError("Unsupported model type")
    return model

def test_model(model, data, mode, model_type, model_name, is_grid=False):
    predictions, actuals = [], []
    
    # extract predictions and ground truths
    for samples, targets in tqdm(data, desc='Testing'):
        with torch.no_grad():
            preds = model(samples).detach().cpu()
            # apply softmax to convert logit outputs to probabilities
            if mode == 1:  # MIMO
                preds = softmax(preds, dim=2)
            else:
                preds = softmax(preds, dim=1)
            predictions.append(preds)
            actuals.append(targets)  
    
    # format
    predictions = torch.vstack(predictions).numpy()
    actuals = torch.vstack(actuals).cpu().numpy()
    
    #print(predictions.shape, actuals.shape)
    #print(predictions[:5])
    
    if mode == 1 and predictions.ndim == 3:  # MIMO
        predictions = predictions.mean(axis=1)  # Average over the sequence length
        actuals = actuals.max(axis=1, keepdims=True)
    
    # calc and display metrics
    accuracy, f1 = calculate_metrics(actuals, predictions)
    plot_confusion_matrix(actuals, predictions, [0, 1, 2, 3], model_type, model_name, is_grid=is_grid)
    plot_occupancy_with_time(datamodule.y_train, datamodule.y_test, predictions, model_type, model_name, is_grid=is_grid, is_test=True, sequence_length=params['data']['num_sequence'])
    
    # displaying metrics - regression
    #mse, mae = calculate_metrics_regression(actuals, predictions)
    #return mse, mae
    
    return accuracy, f1

def evaluate_models(grid_dir, model_type):
    results = []

    for model_dir in os.listdir(grid_dir):
        info_path = os.path.join(grid_dir, model_dir, 'hparams.yaml')
        checkpoint_dir = os.path.join(grid_dir, model_dir, 'checkpoints')
        # get the checkpoint (should only be 1)
        filename = os.listdir(checkpoint_dir)[0]
        if filename.endswith(".ckpt"):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            model = load_model(model_type, checkpoint_path)
            #mse, mae = test_model(model, dataloader, mode, model_type)
            #results.append((filename, mse, mae))
            model_name = format_model_name(info_path)
            accuracy, f1 = test_model(model, dataloader, mode, model_type, model_name, is_grid=True)
            results.append((model_name, accuracy, f1))

    #return pd.DataFrame(results, columns=['Model', 'MSE', 'MAE'])
    return pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-Score'])

def format_model_name(info_path):
    # fetch the experiment name from the hparams file for a model - use yaml library
    with open(info_path, 'r') as file:
        params = yaml.safe_load(file)
    return params['experiment_name']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='rnn or lstm')
    parser.add_argument('--checkpoint_path', type=str, default='None', help='Path to the model checkpoint')
    parser.add_argument('--grid_search', type=str, default='False', help='Perform grid search: True or False')
    args = parser.parse_args()
    
    datamodule = RoomOccupancyDataModule(batch_size=1, sequence_length=params['data']['num_sequence'])
    datamodule.setup(stage='test')
    dataloader = datamodule.val_dataloader() # for traditional testing
    #dataloader = datamodule.predict_dataloader() # for prediction on entire dataset (uncomment if doing, and set is_test=False for plotting)
    mode = params['experiment']
    
    if args.grid_search == 'True':
        grid_dir = f'lightning_logs/{args.model_type}/grid_search'
        df_results = evaluate_models(grid_dir, args.model_type)
        display_grid_results(df_results, args.model_type)
    elif args.grid_search == 'False' and os.path.exists(args.checkpoint_path):
        model = load_model(args.model_type, args.checkpoint_path)
        hparams_path = args.checkpoint_path.split('checkpoints')[0] + 'hparams.yaml'
        model_name = format_model_name(hparams_path)
        if test_model(model, dataloader, mode, args.model_type, model_name):
            print('Successfully tested model.')
        else:
            print('Failure in testing model.')
    else:
        print('Please provide a valid grid_search or checkpoint_path argument. Exiting...')