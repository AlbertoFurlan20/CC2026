import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


class TextDataset(Dataset):
    def __init__(self, X, y, max_len):
        self.X = X
        self.y = y
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]
        label = self.y[idx]

        # Pad the text to the maximum length
        padded_text = np.pad(text, (0, self.max_len - len(text)), mode='constant')
        padded_text = torch.tensor(padded_text).long()

        # Convert the label to a tensor
        label = torch.tensor(label).float()

        return padded_text, label


def compute_metrics_for_regression(y_test, y_test_pred):
    metrics = {}
    for task in DIMENSIONS:
        targets_task = [t[DIMENSIONS.index(task)] for t in y_test]
        pred_task = [l[DIMENSIONS.index(task)] for l in y_test_pred]
        
        rmse = mean_squared_error(targets_task, pred_task, squared=False)

        metrics[f"rmse_{task}"] = rmse
    
    return metrics

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def train_model(X_train, y_train, X_valid, y_valid, max_len):
    # Create datasets and data loaders
    train_dataset = TextDataset(X_train, y_train, max_len)
    valid_dataset = TextDataset(X_valid, y_valid, max_len)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Define the model, optimizer, and loss function
    model = LanguageModel(len(set(X_train)), 128, 128, len(DIMENSIONS))
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()

    # Train the model
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
        print(f'Epoch {epoch+1}, Val Loss: {total_loss / len(valid_loader)}')

    return model

def predict(model, X, max_len):
    # Create a dataset and data loader
    dataset = TextDataset(X, None, max_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Make predictions
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch
            inputs = inputs.to('cuda')
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
    return np.array(predictions)

if __name__ == '__main__':

    ellipse_df = pd.read_csv('train.csv', 
                            header=0, names=['text_id', 'full_text', 'Cohesion', 'Syntax', 
                            'Vocabulary', 'Phraseology','Grammar', 'Conventions'], 
                            index_col='text_id')
    ellipse_df = ellipse_df.dropna(axis=0)


    # Process data and store