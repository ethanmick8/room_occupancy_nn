from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import torch
import os
import datetime
from utils.params import get_params

params = get_params()

def calculate_metrics(actuals, predictions):
    """Calculate the accuracy and F1 score of the model predictions. This function
    is used to evaluate the performance of the model on the test set."""
    print(actuals.shape, predictions.shape)
    accuracy = accuracy_score(actuals, np.argmax(predictions, axis=1))
    f1 = f1_score(actuals, np.argmax(predictions, axis=1), average='weighted')
    print(f'Accuracy: {accuracy: .4f}, F1: {f1: .4f}')
    return accuracy, f1

def calculate_metrics_cross_val(actuals, predictions):
    """Cross validation version of calculate_metrics. This function is used to evaluate
    the performance of the model on the test set when using cross validation."""
    # cut off the first training fold since no predictions done on it
    #actuals, predictions = actuals[929:], predictions[929:]
    accuracy = accuracy_score(actuals, predictions)
    f1_mac = f1_score(actuals, predictions, average='macro')
    f1_weighted = f1_score(actuals, predictions, average='weighted')
    return accuracy, f1_mac, f1_weighted

def plot_confusion_matrix(actuals, predictions, classes, model_type, model_name, is_grid=False, is_cross_val=False):
    """_summary_ This function plots the confusion matrix of the model's predictions
    on the test set. It is used to visualize the performance of the model on the test set.

    Args:
        actuals (_type_): _description_ The actual labels of the test set
        predictions (_type_): _description_ The predicted labels of the test set
        classes (_type_): _description_ The classes of the dataset
        model_type (_type_): _description_ The type of model used
        model_name (_type_): _description_ The name of the model
        is_grid (bool, optional): _description_. Defaults to False. Specifies whether or not dealing w/ grid search
        is_cross_val (bool, optional): _description_. Defaults to False. Same as above but cross val
    """
    if not is_cross_val: # if not cross validation, convert softmax probabilities to class labels
        cm = confusion_matrix(actuals, np.argmax(predictions, axis=1), labels=classes)
    else: # if cross validation, predictions are already class labels
        cm = confusion_matrix(actuals, predictions, labels=classes)
    # sklearn confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    num_folds = model_name.split('_')[-1]
    if model_type == 'svm':
        plt.title(f'{model_type} Confusion Matrix - {num_folds} Cross Validation - {params["svm"]["kernel"]} Kernel - C={params["svm"]["C"]}')
    else:
        plt.title(f'{model_type} Confusion Matrix - {num_folds} Cross Validation')
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if is_grid: # save to grid search folder - multiple models
        output_dir = f'figures/{model_type}/grid_search/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/CM_{date}.png')
    else: # save to single model folder
        output_dir = f'figures/{model_type}/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/CM_{date}.png')
        plt.show()
    
def plot_occupancy_with_time(y_train, y_test, predictions, model_type, model_name, is_grid=False, is_test=True, is_cross_val=False, sequence_length=25):
    """_summary_

    Args:
        y_train (_type_): _description_ The training data
        y_test (_type_): _description_ The test data
        predictions (_type_): _description_ The model's predictions
        model_type (_type_): _description_ The type of model used
        model_name (_type_): _description_ The name of the model
        is_grid (bool, optional): _description_. Defaults to False.
        is_test (bool, optional): _description_. Defaults to True. Specifies whether or not dealing w/ test set (False means predict
            dataloader is utilized (unorthodox method followed in paper for confusion matrices))
        is_cross_val (bool, optional): _description_. Defaults to False.
        sequence_length (int, optional): _description_. Defaults to 25. The length of the input sequences

    Raises:
        ValueError: _description_ If predictions and prediction time steps do not have the same length

    Returns:
        _type_: _description_ None, both saves and displays the plot
    """
    plt.figure(figsize=(12, 6))

    if predictions.ndim == 3:  # Check if predictions are 3D - MIMO
        predictions = predictions.mean(axis=1)  # Average over the sequence length
    if predictions.ndim > 2: # convert softmax probabilities to class labels
        predictions = np.argmax(predictions, axis=2).mean(axis=1)
    else:
        # if cross validation, predictions are already class labels
        if not is_cross_val:
            predictions = np.argmax(predictions, axis=1)

    if y_train.shape == y_test.shape: # cross validation
        total_time_steps = range(len(y_train))
        if is_test:
            return ValueError("is_test must be False for cross validation")
        prediction_time_steps = total_time_steps
    else:
        # Generating time steps for training and testing
        train_time_steps = range(len(y_train))
        test_time_steps = range(len(y_train), len(y_train) + len(y_test))
        # Ensure sizes match
        if is_test:
            prediction_time_steps = range(len(y_train) + sequence_length - 1, len(y_train) + len(y_test))
        else:
            prediction_time_steps = range(sequence_length - 1, len(y_train) + len(y_test))
        if len(predictions) != len(prediction_time_steps):
            raise ValueError(f"Predictions and prediction time steps must have the same length. Their lengths are: {len(predictions)} and {len(prediction_time_steps)}")
    # Plotting training data
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, np.ndarray):
        if y_train.shape == y_test.shape:
            plt.plot(total_time_steps, y_train, label='Ground Truths', color='blue')
        else:
            plt.plot(train_time_steps, y_train, label='Training Data', color='blue')
    else:
        raise ValueError("y_train must be a pandas DataFrame")

    # Plotting test data
    if isinstance(y_test, pd.DataFrame)  or isinstance(y_train, np.ndarray):
        if y_train.shape != y_test.shape:
            plt.plot(test_time_steps, y_test, label='Test Data', color='red')
    else:
        raise ValueError("y_test must be a pandas DataFrame")

    # Plotting predictions
    plt.plot(prediction_time_steps, predictions, label='Predictions', color='green', linestyle='--')
    num_folds = model_name.split('_')[-1]
    if model_type == 'svm':
        plt.title(f'{model_type} Occupancy Count v Time: {num_folds} Cross Validation - {params["svm"]["kernel"]} Kernel - C={params["svm"]["C"]}')
    else:
        plt.title(f'{model_type} - Occupancy Count v Time: {num_folds} Cross Validation')
    plt.xlabel('Time Index')
    plt.ylabel('Occupancy Count (Target)')
    plt.legend()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if is_grid: # save to grid search folder - multiple models
        output_dir = f'figures/{model_type}/grid_search/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/occupancy_v_time_{date}.png')
    else: # save to single model folder
        output_dir = f'figures/{model_type}/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/occupancy_v_time_{date}.png')
        plt.show()

def display_grid_results(df_results, model_type):
    """
    Display the model evaluation results in a formatted table akin to those in the paper.
    
    Args:
    df_results (DataFrame): A pandas DataFrame containing the columns 'Model', 'Accuracy', and 'F1-Score
    """
    # sort by F1-Score
    sorted_df = df_results.sort_values(by='F1-Score', ascending=False)
    print(sorted_df)
    
    # save to csv
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'figures/{model_type}/grid_search'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sorted_df.to_csv(f'{output_dir}/{date}.csv')