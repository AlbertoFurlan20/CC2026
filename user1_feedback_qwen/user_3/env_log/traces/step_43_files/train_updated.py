import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
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
train_df = pd.read_csv('/path/to/train.csv')
val_df = pd.read_csv('/path/to/val.csv')
test_df = pd.read_csv('/path/to/test.csv')

# Preprocess the text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create dataset and dataloaders
train_dataset = TextDataset(train_df['text'].values, train_df['label'].values, tokenizer, max_len=512)
val_dataset = TextDataset(val_df['text'].values, val_df['label'].values, tokenizer, max_len=512)
test_dataset = TextDataset(test_df['text'].values, test_df['label'].values, tokenizer, max_len=512)

train_dataloader = DataLoader(train_dataset, batch_size=2)
val_dataloader = DataLoader(val_dataset, batch_size=2)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Define a complex neural network model for regression
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[1]
        dropout_output = self.dropout(pooled_output)
        fc1_output = nn.ReLU()(self.fc1(dropout_output))
        fc2_output = nn.ReLU()(self.fc2(fc1_output))
        fc3_output = nn.ReLU()(self.fc3(fc2_output))
        reg_output = self.fc4(fc3_output)
        return reg_output.squeeze(-1)

# Initialize the model, optimizer, and loss function
model = RegressionModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # Number of epochs
    model.train()
    for batch in train_dataloader:
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
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_dataloader)
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')

print("Training complete.")

# Save the trained model
torch.save(model.state_dict(), 'regression_model.pth')

# Generate predictions for the test set
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].