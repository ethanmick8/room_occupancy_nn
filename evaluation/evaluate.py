import torch
import numpy as np
import argparse
from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from pytorch_lightning import Trainer
from tqdm import tqdm
import joblib
import os
import pandas as pd

from data.module import RoomOccupancyDataModule
from data.dataset import RoomOccupancyDataset
from utils.metrics import calculate_metrics, plot_predictions, plot_feature_with_time, plot_occupancy_with_time, display_grid_results
from utils.params import get_params

params = get_params()

def load_model(model_type, checkpoint_path):
    # dynamically load the model from checkpoint
    if model_type == 'rnn':
        model = EJM_RNN.load_from_checkpoint(checkpoint_path)
    elif model_type == 'lstm':
        model = EJM_LSTM.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError("Unsupported model type")
    return model

def test_model(model, data, mode, model_type):
    predictions, actuals = [], []
    # extract predictions and ground truths
    for samples, targets in tqdm(data, desc='Testing'):
        with torch.no_grad():
            
            if mode == 0: # MISO
                preds = model(samples).detach().cpu()
            else: # MIMO
                preds = model(samples, targets.size()[1]).detach().cpu()
                
        predictions.append(preds)
        actuals.append(targets)    
    
    # format
    predictions = torch.vstack(predictions).cpu().numpy()
    actuals = torch.vstack(actuals).cpu().numpy()
    
    #print(predictions.shape, actuals.shape)
    
    # displaying metrics
    mse, mae = calculate_metrics(actuals, predictions)
    #plot_predictions(actuals, predictions, type(model).__name__)
    
    # plotting
    plot_occupancy_with_time(datamodule.y_train, datamodule.y_test, predictions, model_type)
    
    # various parts of code used when predicting individual features
    '''# load in the scaler
    scaler = joblib.load('scaler.pkl')
    
    # separate the binary and numeric columns
    params = get_params()
    predictions_numeric = predictions[:, :16]
    predictions_binary = predictions[:, 16:]
    
    # unscale numerics and recombine with binary
    predictions_numeric_unscaled = scaler.inverse_transform(predictions_numeric)
    #print(predictions_numeric_unscaled.shape)
    predictions_unscaled = np.concatenate((predictions_numeric_unscaled, predictions_binary), axis=1)
    
    #print(predictions.shape)
    #print(actuals.shape, predictions_unscaled.shape)
    
    # displaying metrics
    #calculate_metrics(actuals, predictions_unscaled)
    #plot_predictions(actuals, predictions_unscaled, type(model).__name__)

    # plotting - single feature way
    #predictions_unscaled = predictions_unscaled[:, 10]
    #print(predictions_unscaled)
    #plot_feature_with_time(datamodule.X_train, datamodule.X_test, predictions_unscaled, 'S1_Sound', model_type)'''
    
    return mse, mae

def evaluate_models(checkpoint_dir, model_type):
    
    results = []

    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".ckpt"):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            model = load_model(model_type, checkpoint_path)
            mse, mae = test_model(model, dataloader, mode, model_type)
            results.append((filename, mse, mae))

    return pd.DataFrame(results, columns=['Model', 'MSE', 'MAE'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='rnn or lstm')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()
    
    datamodule = RoomOccupancyDataModule(batch_size=1, sequence_length=params['data']['num_sequence'])
    datamodule.setup(stage='test')
    dataloader = datamodule.val_dataloader()
    mode = get_params()['experiment']
    df_results = evaluate_models(args.checkpoint_dir, args.model_type)
    display_grid_results(df_results)
    
    '''model = load_model(args.model_type, args.checkpoint_path)
    datamodule = RoomOccupancyDataModule(batch_size=1, sequence_length=params['data']['num_sequence'])
    datamodule.setup(stage='test')
    dataloader = datamodule.val_dataloader()

    mode = get_params()['experiment']

    if test_model(model, dataloader, mode, args.model_type):
        print('Successfully tested model.')
    else:
        print('Failure in testing model.')'''