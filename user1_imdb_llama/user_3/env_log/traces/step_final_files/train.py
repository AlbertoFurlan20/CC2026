from datasets import load_dataset
import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    
    # Import the IMDb dataset from the Hugging Face library
    imdb = load_dataset("stanfordnlp/imdb")

    # Split the dataset into training and testing sets
    train_dataset = imdb["train"]
    test_dataset = imdb["test"]

    # Define the model by instantiating the DistilBERT model and adding a classification head
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Define a custom dataset class for our dataset
    class IMDBDataset(Dataset):
        def __init__(self, dataset, tokenizer):
            self.dataset = dataset
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            text = self.dataset[idx]["text"]
            label = self.dataset[idx]["label"]

            encoding = self.tokenizer.encode_plus(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Initialize the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Create dataset instances
    train_dataset = IMDBDataset(train_dataset, tokenizer)
    test_dataset = IMDBDataset(test_dataset, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Define the training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, dim=1)
                predictions.extend(predicted.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())

        accuracy = accuracy_score(labels, predictions)
        print(f'Epoch {epoch+1}, Test Accuracy: {accuracy}')

    # Save the model and the test set predictions to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(test_dataset)))
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            text = data["text"]
            label = data["label"]
            encoding = tokenizer.encode_plus(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].flatten().unsqueeze(0).to(device)
            attention_mask = encoding['attention_mask'].flatten().unsqueeze(0).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.softmax(logits, dim=1)
            submission.loc[idx] = pred.tolist()[0]

    submission.to_csv('submission.csv', index_label='idx')