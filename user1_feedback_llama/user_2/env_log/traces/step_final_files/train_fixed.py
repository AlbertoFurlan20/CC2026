import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_regression(y_test, y_test_pred):
    metrics = {}
    for task in DIMENSIONS:
        targets_task = [t[DIMENSIONS.index(task)] for t in y_test]
        pred_task = [l[DIMENSIONS.index(task)] for l in y_test_pred]
        
        rmse = mean_squared_error(targets_task, pred_task, squared=False)

        metrics[f"rmse_{task}"] = rmse
    
    return metrics

def train_model(X_train, y_train, X_valid, y_valid):
    # Define the model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(1000, 500)  # input layer (1000) -> hidden layer (500)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(500, 250)  # hidden layer (500) -> hidden layer (250)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(250, 8)  # hidden layer (250) -> output layer (8)
        
        def forward(self, x):
            out = self.relu1(self.fc1(x))
            out = self.relu2(self.fc2(out))
            out = self.fc3(out)
            return out
    
    model = NeuralNetwork()
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    X_valid = torch.tensor(X_valid).to(device)
    y_valid = torch.tensor(y_valid).to(device)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                valid_outputs = model(X_valid)
                valid_loss = criterion(valid_outputs, y_valid)
                print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Valid Loss: {valid_loss.item():.4f}')
            model.train()
    
    return model

def predict(model, X):
    # Convert the input to a tensor
    X = torch.tensor(X)
    
    # Move the input to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(X)
    
    # Convert the output to a numpy array
    outputs = outputs.cpu().numpy()
    
    return outputs

if __name__ == '__main__':

    ellipse_df = pd.read_csv('train.csv', 
                            header=0, names=['text_id', 'full_text', 'Cohesion', 'Syntax', 
                            'Vocabulary', 'Phraseology','Grammar', 'Conventions'], 
                            index_col='text_id')
    ellipse_df = ellipse_df.dropna(axis=0)


    # Process data and store into numpy arrays.
    data_df = ellipse_df
    X = list(data_df.full_text.to_numpy())
    y = np.array([data_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X))])

    # Create a train-valid split of the data.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # define and train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # evaluate the model on the valid set using compute_metrics_for_regression and print the results
    # should fill out the predict function
    y_valid_pred = predict(model, X_valid)
    metrics = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(metrics)
    print("final MCRMSE on validation set: ", np.mean(list(metrics.values())))

    # save submission.csv file for the test set
    submission_df = pd.read_csv('test.csv