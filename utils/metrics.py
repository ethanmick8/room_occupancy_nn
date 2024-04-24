from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import torch
import os
import datetime

from utils.params import get_params

def calculate_metrics(actuals, predictions):
    accuracy = accuracy_score(actuals, np.argmax(predictions, axis=1))
    f1 = f1_score(actuals, np.argmax(predictions, axis=1), average='weighted')
    print(f'Accuracy: {accuracy: .4f}, F1: {f1: .4f}')
    return accuracy, f1

def plot_confusion_matrix(actuals, predictions, classes, model_type, model_name, is_grid=False):
    cm = confusion_matrix(actuals, np.argmax(predictions, axis=1), labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_type} Confusion Matrix')
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if is_grid:
        output_dir = f'figures/{model_type}/grid_search/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/CM_{date}.png')
    else:
        output_dir = f'figures/{model_type}/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/CM_{date}.png')
        plt.show()

def calculate_metrics_regression(actuals, predictions):
    params = get_params()
    if params['experiment'] == 1: # MIMO
        actuals = actuals.mean(axis=1)
        predictions = predictions.mean(axis=1)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f'MSE: {mse: .4f}, MAE: {mae: .4f}')
    return mse, mae
    
def plot_occupancy_with_time(y_train, y_test, predictions, model_type, model_name, is_grid=False, sequence_length=25):
    plt.figure(figsize=(12, 6))

    #if predictions.ndim == 3:  # Check if predictions are 3D - MIMO
    #    predictions = predictions.mean(axis=1)  # Average over the sequence length
    if predictions.ndim > 2: # convert softmax probabilities to class labels
        predictions = np.argmax(predictions, axis=2).mean(axis=1)
    else:
        predictions = np.argmax(predictions, axis=1)

    # Generating time steps for training and testing
    train_time_steps = range(len(y_train))
    test_time_steps = range(len(y_train), len(y_train) + len(y_test))

    # Ensure sizes match
    prediction_time_steps = range(len(y_train) + sequence_length - 1, len(y_train) + len(y_test))
    if len(predictions) != len(prediction_time_steps):
        raise ValueError(f"Predictions and test time steps must have the same length. Their lengths are: {len(predictions)} and {len(prediction_time_steps)}")

    # Plotting training data
    if isinstance(y_train, pd.DataFrame):
        plt.plot(train_time_steps, y_train, label='Training Data', color='blue')
    else:
        raise ValueError("y_train must be a pandas DataFrame")

    # Plotting test data
    if isinstance(y_test, pd.DataFrame):
        plt.plot(test_time_steps, y_test, label='Test Data', color='red')
    else:
        raise ValueError("y_test must be a pandas DataFrame")

    # Plotting predictions
    plt.plot(prediction_time_steps, predictions, label='Predictions', color='green', linestyle='--')
    plt.title(f'{model_type} - Occupancy Count v Time w/ Predictions')
    plt.xlabel('Time Index')
    plt.ylabel('Occupancy Count (Target)')
    plt.legend()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if is_grid:
        output_dir = f'figures/{model_type}/grid_search/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/occupancy_v_time_{date}.png')
    else:
        output_dir = f'figures/{model_type}/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/occupancy_v_time_{date}.png')
        plt.show()

def display_grid_results(df_results, model_type):
    """
    Display the model evaluation results in a formatted table.
    
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

  
#X_train, X_test, y_train, y_test = fetch_and_split_data()

#plot_feature_with_time(X_train, X_test, 'S3_Temp')