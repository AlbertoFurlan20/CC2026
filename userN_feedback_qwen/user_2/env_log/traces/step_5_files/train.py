import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import random
import os

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=0, names=['text_id', 'full_text', 'Cohesion', 'Syntax', 'Vocabulary', 'Phraseology', 'Grammar', 'Conventions'], index_col='text_id')
    test_df = pd.read_csv(test_path, header=0, names=['text_id', 'full_text'], index_col='text_id')

    train_df = train_df.dropna(axis=0)
    test_df = test_df.dropna(axis=0)

    train_texts = train_df['full_text'].tolist()
    train_labels = train_df.drop(['full_text'], axis=1).values

    test_texts = test_df['full_text'].tolist()
    test_labels = np.zeros((len(test_texts), len(DIMENSIONS)))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_len = 512

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)

    return train_dataset, test_dataset


def train_model(train_loader, valid_loader, model, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

    best_valid_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}')


def predict(model, test_loader, device):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    model.to(device)

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.log