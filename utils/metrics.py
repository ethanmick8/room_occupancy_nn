from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import torch
import datetime

def calculate_metrics(actuals, predictions):
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f'MSE: {mse: .4f}, MAE: {mae: .4f}')

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
    plt.savefig(f'figures/{feature_name}_vs_time_with_predictions_{date}.png')
    plt.show()
  
#X_train, X_test, y_train, y_test = fetch_and_split_data()

#plot_feature_with_time(X_train, X_test, 'S3_Temp')