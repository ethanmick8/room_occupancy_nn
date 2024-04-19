from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import datetime
from datasets.dataset import RoomOccupancyDataset, fetch_and_split_data

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

def plot_feature_with_time(X_train, X_test, predictions, feature_name):
    plt.figure(figsize=(12, 6))
    # Plotting training data
    train_time_steps = range(len(X_train))
    plt.plot(train_time_steps, X_train[feature_name], label='Training Data', color='blue')
    # Plotting test data
    test_time_steps = range(len(X_train), len(X_train) + len(X_test))
    plt.plot(test_time_steps, X_test[feature_name], label='Test Data', color='orange')
    # Plotting predictions
    plt.plot(test_time_steps, predictions, label='Predictions', color='green', linestyle='--')
    plt.title(f'Time Series Plot for {feature_name} with Predictions')
    plt.xlabel('Time Index')
    plt.ylabel(feature_name)
    plt.legend()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(f'figures/{feature_name}_vs_time_with_predictions_{date}.png')
    #plt.show()
  
#X_train, X_test, y_train, y_test = fetch_and_split_data()

#plot_feature_with_time(X_train, X_test, 'S3_Temp')