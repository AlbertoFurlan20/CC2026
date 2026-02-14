import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForPreTraining
import pandas as pd

# Define a custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load dataset from CSV files (example: using pandas)
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# Preprocess the text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the target labels for the six analytic measures
target_labels = ['measure1', 'measure2', 'measure3', 'measure4', 'measure5', 'measure6']

# Create dataset and dataloaders
train_datasets = [TextDataset(train_df['text'].values, train_df[target].values, tokenizer, max_len=512) for target in target_labels]
val_datasets = [TextDataset(val_df['text'].values, val_df[target].values, tokenizer, max_len=512) for target in target_labels]

train_dataloaders = [DataLoader(dataset, batch_size=2) for dataset in train_datasets]
val_dataloaders = [DataLoader(dataset, batch_size=2) for dataset in val_datasets]

# Define a simple neural network model for regression
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.bert = BertForPreTraining.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.reg_head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[1]
        dropout_output = self.dropout(pooled_output)
        reg_output = self.reg_head(dropout_output)
        return reg_output.squeeze(-1)

# Initialize the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_val_losses = [float('inf')] * len(target_labels)

for i, target in enumerate(target_labels):
    model = RegressionModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(3):  # Number of epochs
        model.train()
        for batch in train_dataloaders[i]:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloaders[i]:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloaders[i])
        if val_loss < best_val_losses[i]:
            best_val_losses[i] = val_loss
            torch.save(model.state_dict(), f'regression_model_{target}.pth')

    print(f'Target: {target}, Best Validation Loss: {best_val_losses[i]}')

# Save the trained model and predictions for the test set
test_dataset = TextDataset(test_df['text'].values, test_df[target].values, tokenizer, max_len=512)
test_dataloader = DataLoader(test_dataset, batch_size=2)

for i, target in enumerate(target_labels):
    model = RegressionModel().