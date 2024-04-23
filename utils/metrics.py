from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import torch
import datetime

from utils.params import get_params

def calculate_metrics(actuals, predictions):
    params = get_params()
    if params['experiment'] == 1: # MIMO
        actuals = actuals.mean(axis=1)
        predictions = predictions.mean(axis=1)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f'MSE: {mse: .4f}, MAE: {mae: .4f}')
    return mse, mae

def plot_predictions(actuals, predictions, model_type):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:200], label='Actual')
    plt.plot(predictions[:200], label='Predicted')
    plt.title(f'{model_type}: Predictions vs Actuals')
    plt.legend()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(f'figures/{model_type}_predictions_{date}.png')


def plot_feature_with_time(X_train, X_test, predictions, feature_name, model_type, sequence_length=25):
    plt.figure(figsize=(12, 6))
    
    # Generating time steps for training and testing
    train_time_steps = range(len(X_train))
    test_time_steps = range(len(X_train), len(X_train) + len(X_test) - sequence_length + 1)
    
    # ensure sizes match
    if len(predictions) != len(test_time_steps):
        raise ValueError(f"Predictions and test time steps must have the same length. Their lengths are: {len(predictions)} and {len(test_time_steps)}")

    # Plotting training data
    if isinstance(X_train, pd.DataFrame):
        plt.plot(train_time_steps, X_train[feature_name], label='Training Data', color='blue')
    else:
        raise ValueError("X_train must be a pandas DataFrame")

    test_time_steps_2 = range(len(X_train), len(X_train) + len(X_test))

    # Plotting test data
    if isinstance(X_test, pd.DataFrame):
        plt.plot(test_time_steps_2, X_test[feature_name], label='Test Data', color='orange')
    else:
        raise ValueError("X_test must be a pandas DataFrame")

    # Plotting predictions
    plt.plot(test_time_steps, predictions, label='Predictions', color='green', linestyle='--')
    plt.title(f'{model_type} - {feature_name} over time with Predictions')
    plt.xlabel('Time Index')
    plt.ylabel(feature_name)
    plt.legend()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(f'figures/{model_type}/{feature_name}_vs_time_with_predictions_{date}.png')
    plt.show()
    
# similar to previous function, but instead concerned with actual target (occupancy count)
def plot_occupancy_with_time(y_train, y_test, predictions, model_type, sequence_length=25):
    plt.figure(figsize=(12, 6))

    if predictions.ndim == 3:  # Check if predictions are 3D - MIMO
        predictions = predictions.mean(axis=1)  # Average over the sequence length

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
    plt.title(f'{model_type} - Occupancy Count over Time with Predictions')
    plt.xlabel('Time Index')
    plt.ylabel('Occupancy Count (Target)')
    plt.legend()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(f'figures/{model_type}/target_performance_over_time_{date}.png')
    plt.show()

def display_grid_results(df_results):
    """
    Display the model evaluation results in a formatted table.
    
    Args:
    df_results (DataFrame): A pandas DataFrame containing the columns 'Model' and 'MSE'.
    """
    # Use pandas styling to highlight the minimum values in the MSE column
    styled_df = df_results.style.highlight_min(subset=['MSE'], color='lightgreen', axis=0)
    
    # Display the styled DataFrame
    print(styled_df.to_string())

    # sort by best
    sorted_df = df_results.sort_values('MSE', ascending=True).reset_index(drop=True)
    print(sorted_df)

  
#X_train, X_test, y_train, y_test = fetch_and_split_data()

#plot_feature_with_time(X_train, X_test, 'S3_Temp')