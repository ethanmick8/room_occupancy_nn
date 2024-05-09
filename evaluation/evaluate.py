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
from utils.metrics import calculate_metrics, calculate_metrics_cross_val, plot_confusion_matrix, plot_occupancy_with_time, display_grid_results
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

def evaluate_cross_validation(model_type, folds_dir, num_folds=10, data_length=10129):
    datamodule_cv = RoomOccupancyDataModule(batch_size=1, sequence_length=params['data']['num_sequence'], num_splits=num_folds, is_cross_val=True)
    
    # Initialize an array to store the final predictions
    final_predictions = np.full(data_length, np.nan)
    print(f'Shape of final predictions: {final_predictions.shape}')
    train_checkpoint_path = ''
    # Process each fold
    for fold in tqdm(range(num_folds)):
        datamodule_cv.current_fold = fold
        datamodule_cv.setup(stage='test')

        checkpoint_dir = os.path.join(folds_dir, f'fold_{fold}', 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
        if fold == 0:
            train_checkpoint_path = checkpoint_path
        model = load_model(model_type, checkpoint_path)

        data_loaders = datamodule_cv.get_fold_dataloader(fold) # get the dataloaders for the current fold
        val_loader = data_loaders[1] # validation loader for the current fold
        fold_predictions, actuals = [], []
        for samples, targets in val_loader:
            with torch.no_grad():
                preds = model(samples).detach().cpu()
                preds = softmax(preds, dim=1 if params['experiment'] != 1 else 2)
                fold_predictions.append(preds)
                actuals.append(targets)
        
        fold_predictions = np.argmax(torch.vstack(fold_predictions).numpy(), axis=1)
        # grab og indices from the datamodule
        original_indices = datamodule_cv.validation_indices[fold]
        # align with length of fold predictions (sequence length causes misalignment)
        aligned_indices = original_indices[:len(fold_predictions)]
        remaining_indices = original_indices[len(fold_predictions):]
        # place predictions accordingly; if there are any remaining indices, use the last prediction
        final_predictions[aligned_indices] = fold_predictions
        if len(remaining_indices) > 0:
            final_predictions[remaining_indices] = fold_predictions[-1]

    # setting first training fold predictions to NaN - (alternative option)
    #final_predictions = np.nan_to_num(final_predictions)
    
    # ignore values for the first training fold because no predictions were made
    nan_indices = np.argwhere(np.isnan(final_predictions)).flatten()
    # cut off the first training fold since no predictions done on it
    actuals = datamodule_cv.y[len(nan_indices):]
    final_predictions = final_predictions[len(nan_indices):]
    print(f'Actuals shape: {actuals.shape}, Predictions shape: {final_predictions.shape}')
    

    # Calculate metrics and plot results after all folds are processed
    #actuals = datamodule_cv.y
    actuals = actuals.to_numpy()
    final_predictions = final_predictions.reshape(-1, 1)
    accuracy, f1_mac, f1_weighted = calculate_metrics_cross_val(actuals, final_predictions)
    plot_confusion_matrix(actuals, final_predictions, [0, 1, 2, 3], model_type, f'cross_val_{args.cross_val}-fold', is_grid=False, is_cross_val=True)
    plot_occupancy_with_time(actuals, actuals, final_predictions, model_type, f'cross_val_{args.cross_val}-fold', is_grid=False, is_test=False, is_cross_val=True, sequence_length=params['data']['num_sequence'])

    print(f'Overall Accuracy: {accuracy}, Macro F1-Score: {f1_mac}, Weighted F1-Score: {f1_weighted}')
    return True

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
    plot_occupancy_with_time(datamodule.y_train, datamodule.y_test, predictions, model_type, model_name, is_grid=is_grid, is_test=False, sequence_length=params['data']['num_sequence'])
    
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
    parser.add_argument('--cross_val', type=int, default=0, help='Evaluate using cross-validation results (supply number of folds)')
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
        if args.cross_val > 0:
            if evaluate_cross_validation(args.model_type, args.checkpoint_path, num_folds=args.cross_val):
                print(f'Successfully tested {args.cross_val}-fold cross validated model.')
            else:
                print('Failure in testing cross-validated model.')
        else:
            model = load_model(args.model_type, args.checkpoint_path)
            hparams_path = args.checkpoint_path.split('checkpoints')[0] + 'hparams.yaml'
            model_name = format_model_name(hparams_path)
            if test_model(model, dataloader, mode, args.model_type, model_name):
                print('Successfully tested model.')
            else:
                print('Failure in testing model.')
    else:
        print('Please provide a valid grid_search or checkpoint_path argument. Exiting...')