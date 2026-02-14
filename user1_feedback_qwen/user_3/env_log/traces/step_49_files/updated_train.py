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

# Define the target labels for the six analytic measures
target_labels = ['label1', 'label2', 'label3', 'label4', 'label5', 'label6']

# Create dataset and dataloaders
train_dataset = TextDataset(train_df['text'].values, train_df[target_labels].values, tokenizer, max_len=512)
val_dataset = TextDataset(val_df['text'].values, val_df[target_labels].values, tokenizer, max_len=512)

train_dataloader = DataLoader(train_dataset, batch_size=2)
val_dataloader = DataLoader(val_dataset, batch_size=2)

# Define a complex neural network model for language modeling
class ComplexLanguageModel(nn.Module):
    def __init__(self):
        super(ComplexLanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 6)  # Adjusted for regression task

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[0][:, 0]  # Take the [CLS] token's embedding
        dropout_output1 = self.dropout1(pooled_output)
        fc1_output = nn.ReLU()(self.fc1(dropout_output1))
        dropout_output2 = self.dropout2(fc1_output)
        fc2_output = nn.ReLU()(self.fc2(dropout_output2))
        fc3_output = nn.ReLU()(self.fc3(fc2_output))
        logits = self.fc4(fc3_output)
        return logits

# Initialize the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexLanguageModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()  # Changed to MSELoss for regression task

# Training loop
num_epochs = 3
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}')

    # Evaluation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs