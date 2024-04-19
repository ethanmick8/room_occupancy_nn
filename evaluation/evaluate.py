import torch
import numpy as np
import argparse
from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import RoomOccupancyDataset, fetch_and_split_data
from utils.metrics import calculate_metrics, plot_predictions, plot_feature_with_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fetch the test data
X_train, X_test, _, y_test = fetch_and_split_data()

# some hyperparameters
sequence_length = 25
batch_size = 1

test_dataset = RoomOccupancyDataset(X_test, y_test, sequence_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
input_size = test_loader.dataset.sequences.shape[2]

def test_model(test_loader, model_type, hidden_size=64, num_layers=2):
    # fetch and load the trained model
    model_path = f'checkpoints/trained_model_{model_type}.pkl'
    if model_type == 'rnn':
        model = EJM_RNN(input_size, hidden_size, num_layers)
    elif model_type == 'lstm':
        model = EJM_LSTM(input_size, hidden_size, num_layers)
    else:
        return False
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in tqdm(test_loader, desc='Testing'):
            sequences, targets = sequences.to(device), targets.to(device)
            output = model(sequences)
            # assuming one value per sequence
            predictions.extend(output.cpu().numpy())
            actuals.extend(output.cpu().numpy())
            
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # displaying metrics
    calculate_metrics(actuals, predictions)
    #plot_predictions(actuals, predictions, model_type)
    predictions = predictions.flatten()
    print(predictions)
    plot_feature_with_time(X_train, X_test, predictions, 'S3_Temp')
    
    return True
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='rnn or lstm')
    args = parser.parse_args()
    
    if test_model(test_loader, args.model_type):
        print('Successfully tested model.')
    else:
        print('Invalid model argument.')