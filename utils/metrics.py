from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
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
    plt.savefig(f'../figures/{model_type}_predictions_{date}.png')