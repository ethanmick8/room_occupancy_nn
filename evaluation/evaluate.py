import torch
import numpy as np
import argparse
from models.rnn import EJM_RNN
from models.lstm import EJM_LSTM
from pytorch_lightning import Trainer
from tqdm import tqdm
import joblib

from data.module import RoomOccupancyDataModule
from data.dataset import RoomOccupancyDataset
from utils.metrics import calculate_metrics, plot_predictions, plot_feature_with_time
from utils.params import get_params

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
    
    # load in the scaler
    scaler = joblib.load('scaler.pkl')
    
    # separate the binary and numeric columns
    params = get_params()
    predictions_numeric = predictions[:, :16]
    predictions_binary = predictions[:, 16:]
    
    # unscale numerics and recombine with binary
    predictions_numeric_unscaled = scaler.inverse_transform(predictions_numeric)
    print(predictions_numeric_unscaled.shape)
    predictions_unscaled = np.concatenate((predictions_numeric_unscaled, predictions_binary), axis=1)
    
    print(predictions.shape)
    print(actuals.shape, predictions_unscaled.shape)
    
    # displaying metrics
    #calculate_metrics(actuals, predictions_unscaled)
    #plot_predictions(actuals, predictions_unscaled, type(model).__name__)
    #print(predictions)
    # we want just the S3_Temp feature, which is th 5th column
    predictions_unscaled = predictions_unscaled[:, 10]
    print(predictions_unscaled)
    plot_feature_with_time(datamodule.X_train, datamodule.X_test, predictions_unscaled, 'S1_Sound', model_type)
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', help='rnn or lstm')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()
    
    model = load_model(args.model_type, args.checkpoint_path)
    datamodule = RoomOccupancyDataModule(batch_size=1, sequence_length=25)
    datamodule.setup(stage='test')
    dataloader = datamodule.val_dataloader()

    mode = get_params()['experiment']

    if test_model(model, dataloader, mode, args.model_type):
        print('Successfully tested model.')
    else:
        print('Failure in testing model.')

# old code
'''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        print('Invalid model argument.')'''