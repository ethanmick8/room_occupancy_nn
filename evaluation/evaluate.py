import torch
import numpy as np
import argparse
from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from pytorch_lightning import Trainer
from tqdm import tqdm
import joblib
import pickle
import os
import yaml
import pandas as pd
from torch.nn.functional import softmax
from data.module import RoomOccupancyDataModule
from data.dataset import RoomOccupancyDataset
from utils.metrics import calculate_metrics, calculate_metrics_cross_val, plot_confusion_matrix, plot_occupancy_with_time, display_grid_results
from utils.params import get_params
import datetime

params = get_params() # params master function

def load_model(model_type, checkpoint_path):
    """Load a model from a checkpoint file

    Args:
        model_type (_type_): rnn, lstm, etc.
        checkpoint_path (_type_): Path to the model checkpoint

    Raises:
        ValueError: Unsupported model type

    Returns:
        _type_: The model loaded from the checkpoint
    """
    # dynamically load the model from checkpoint
    if model_type in ['rnn', 'lstm']:
        if model_type == 'rnn':
            model = EJM_RNN.load_from_checkpoint(checkpoint_path)
        elif model_type == 'lstm':
            model = EJM_LSTM.load_from_checkpoint(checkpoint_path)
    elif model_type in ['svm', 'lda']:
        with open(checkpoint_path, 'rb') as file:
            model = pickle.load(file)
    else:
        raise ValueError("Unsupported model type")
    return model
    
def evaluate_cross_validation(model_type, folds_dir, num_folds=10, data_length=10129):
    """ Evaluate a model using cross-validation. This function is dynamic in its ability to
    handle the different model types and cross validation splits present across tests. 
    It concatenates the predictions from each fold and calculates the metrics for the entire dataset,
    analagous to the methods presented in the seminal paper. Plots, confusion matrices, and
    performance metrics are all generated in this beast of a function. Disclaimer: helper
    libraries often refer to classification results as prediction, but in this context, it is
    more accurate to refer to them as classifications. There are no predictions in the traditional sense.
    We deal with classification here.

    Args:
        model_type (str): rnn, lstm, svm, lda, etc.
        folds_dir (str): Path to the directory containing the folds
        num_folds (int, optional): Defaults to 10.
        data_length (int, optional): Defaults to 10129 - length of the dataset

    Returns:
        bool: True if the evaluation was successful, False otherwise
    """
    datamodule_cv = RoomOccupancyDataModule(batch_size=1, 
                                            sequence_length=params['data']['num_sequence'], 
                                            num_splits=num_folds, is_cross_val=True)
    
    # Initialize an array to store the final predictions
    final_predictions = np.full(data_length, np.nan)
    print(f'Shape of final predictions: {final_predictions.shape}')
    
    # Process each fold
    for fold in tqdm(range(num_folds)):
        datamodule_cv.current_fold = fold
        datamodule_cv.setup(stage='test')
        # handle evaluation for SVM and LDA models
        if model_type in ['svm', 'lda']:
            for file in os.listdir(folds_dir):
                # example filename - svm_0.pkl - I need to extract just the 0
                if file.split('.')[0].split('_')[-1] == str(fold):
                    checkpoint_path = os.path.join(folds_dir, file)
                    break
            model = load_model(model_type, checkpoint_path)
            X_val, y_val = datamodule_cv.val_data()  # Fetch validation data for the fold
            # For SVM/LDA, data should be properly formatted for direct inference
            predictions = model.predict(X_val)
            original_indices = datamodule_cv.validation_indices[fold]
            aligned_length = min(len(predictions), len(original_indices))
            final_predictions[original_indices[:aligned_length]] = predictions[:aligned_length]
            print(f"Fold {fold + 1}/{num_folds}, Predictions: {len(predictions)}, Target Indices: {len(original_indices)}")
        else:
            checkpoint_dir = os.path.join(folds_dir, f'fold_{fold}', 'checkpoints')
            checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
            model = load_model(model_type, checkpoint_path)
            # For RNN/LSTM, use the model in a PyTorch framework to predict
            val_loader = datamodule_cv.get_fold_dataloader(fold)[1]  # validation loader for the current fold
            predictions, y_val = [], []
            for samples, targets in val_loader:
                with torch.no_grad():
                    preds = model(samples).detach().cpu()
                    preds = softmax(preds, dim=1 if params['experiment'] != 1 else 2)
                    predictions.extend(torch.argmax(preds, dim=1).numpy())
                    y_val.extend(targets.numpy())

            # Store fold predictions in the final predictions array
            original_indices = datamodule_cv.validation_indices[fold]
            final_predictions[original_indices] = predictions

    # Handle NaN values if any remain
    nan_indices = np.isnan(final_predictions)
    actuals = datamodule_cv.y[~nan_indices]
    final_predictions = final_predictions[~nan_indices]

    print(f'Actuals shape: {actuals.shape}, Predictions shape: {final_predictions.shape}')

    # Calculate metrics and plot results after all folds are processed
    accuracy, f1_mac, f1_weighted = calculate_metrics_cross_val(actuals, final_predictions)
    plot_confusion_matrix(actuals, final_predictions, [0, 1, 2, 3], model_type, f'cross_val_{num_folds}-fold', is_grid=False, is_cross_val=True)
    plot_occupancy_with_time(actuals, actuals, final_predictions, model_type, f'cross_val_{num_folds}-fold', is_grid=False, is_test=False, is_cross_val=True, sequence_length=params['data']['num_sequence'])
    # saving metric results to csv
    results = pd.DataFrame({'Accuracy': [accuracy], 'F1-Macro': [f1_mac], 'F1-Weighted': [f1_weighted]})
    results.to_csv(f'figures/{model_type}/cross_val_{num_folds}-fold/results.csv', index=False)
    print(f'Overall Accuracy: {accuracy}, Macro F1-Score: {f1_mac}, Weighted F1-Score: {f1_weighted}')
    return True


def test_model(model, data, mode, model_type, model_name, is_grid=False):
    """Evaluate a model on a test set. This function is used when cross-validation is not used.
    It calculates the accuracy and F1-score of the model on the test set, and plots the confusion 
    matrix and occupancy with time. (With help from the metrics.py file)"""
    predictions, actuals = [], []
    
    # extract predictions and ground truths
    for samples, targets in tqdm(data, desc='Testing'):
        with torch.no_grad():
            preds = model(samples).detach().cpu()
            # apply softmax to convert logit outputs to probabilities
            if mode == 1:  # MIMO
                preds = softmax(preds, dim=2)
            else: # different dimension for MISO and SISO
                preds = softmax(preds, dim=1)
            predictions.append(preds)
            actuals.append(targets)  
    
    # format
    predictions = torch.vstack(predictions).numpy()
    actuals = torch.vstack(actuals).cpu().numpy()
    
    if mode == 1 and predictions.ndim == 3:  # MIMO
        predictions = predictions.mean(axis=1)  # Average over the sequence length
        actuals = actuals.max(axis=1, keepdims=True)
    
    # calc and display metrics
    accuracy, f1 = calculate_metrics(actuals, predictions)
    plot_confusion_matrix(actuals, predictions, [0, 1, 2, 3], model_type, model_name, is_grid=is_grid)
    plot_occupancy_with_time(datamodule.y_train, datamodule.y_test, predictions, model_type, model_name, 
                             is_grid=is_grid, is_test=False, sequence_length=params['data']['num_sequence'])
    
    return accuracy, f1

def evaluate_models(grid_dir, model_type):
    """Evaluate models from a grid search - This function is analogous to test_model,
    but it is used specifically to perform comprehensive evaluation on all configurations of
    grid search expected to have been saved accordingly from the train script.

    Args:
        grid_dir (_type_): Path to the grid search directory
        model_type (_type_): rnn, lstm, etc.

    Returns:
        _type_: A DataFrame containing the results of the grid search
    """
    results = []

    for model_dir in os.listdir(grid_dir):
        info_path = os.path.join(grid_dir, model_dir, 'hparams.yaml')
        checkpoint_dir = os.path.join(grid_dir, model_dir, 'checkpoints')
        # get the checkpoint (should only be 1)
        filename = os.listdir(checkpoint_dir)[0]
        if filename.endswith(".ckpt"):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            model = load_model(model_type, checkpoint_path)
            model_name = format_model_name(info_path)
            accuracy, f1 = test_model(model, dataloader, mode, model_type, model_name, is_grid=True)
            results.append((model_name, accuracy, f1))

    #return pd.DataFrame(results, columns=['Model', 'MSE', 'MAE'])
    return pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-Score'])

def format_model_name(info_path):
    """fetch the experiment name from the hparams file for a model - use yaml library
    from the logs"""
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
    # the above is necessary as this variability was not directly implemented into the functionality of the script. Not enough time..
    mode = params['experiment']
    # evaluate parameters to determine what kind of evaluation to perform - grid search or single model -- cross-validation
    if args.grid_search == 'True':
        # no cross-validation for grid search -- confusion matrices generated so staying in line with paper implementation
        grid_dir = f'lightning_logs/{args.model_type}/grid_search'
        df_results = evaluate_models(grid_dir, args.model_type)
        display_grid_results(df_results, args.model_type)
    elif args.grid_search == 'False' and os.path.exists(args.checkpoint_path):
        if args.cross_val > 0: # cross-validation evaluation
            if evaluate_cross_validation(args.model_type, args.checkpoint_path, num_folds=args.cross_val):
                print(f'Successfully tested {args.cross_val}-fold cross validated model.')
            else:
                print('Failure in testing cross-validated model.')
        else: # single model evaluation
            model = load_model(args.model_type, args.checkpoint_path)
            hparams_path = args.checkpoint_path.split('checkpoints')[0] + 'hparams.yaml'
            model_name = format_model_name(hparams_path)
            if test_model(model, dataloader, mode, args.model_type, model_name):
                print('Successfully tested model.')
            else:
                print('Failure in testing model.')
    else:
        print('Please provide a valid grid_search or checkpoint_path argument. Exiting...')