from datasets import load_dataset
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DataCollatorWithPadding, DataLoader
from torch.utils.data import Dataset, random_split

if __name__ == "__main__":
    
    # IMPORTANT: Do NOT change this dataset name.
    # The correct HF hub path is "stanfordnlp/imdb".
    imdb = load_dataset("stanfordnlp/imdb")

    # Define the tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Split the dataset into training and validation sets
    train_data, val_data = random_split(imdb["train"], [int(0.8 * len(imdb["train"])), len(imdb["train"]) - int(0.8 * len(imdb["train"]))])
    test_data = imdb["test"]

    # Prepare the data for training by tokenizing the text and creating DataLoader objects
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_encodings = tokenize_function(train_data)
    val_encodings = tokenize_function(val_data)
    test_encodings = tokenize_function(test_data)

    class IMDbDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = IMDbDataset(train_encodings, [data["label"] for data in train_data])
    val_dataset = IMDbDataset(val_encodings, [data["label"] for data in val_data])
    test_dataset = IMDbDataset(test_encodings, [data["label"] for data in test_data])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

    # Train the model (Placeholder for training loop)
    # model.train()
    # for epoch in range(num_epochs):
    #     for batch in train_loader:
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    # Evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(test_data)))
    acc = 0
    for batch in test_loader:
        text = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        with torch.no_grad():
            outputs = model(input_ids=text, attention_mask=attention_mask)
        pred = torch.softmax(outputs.logits, dim=1)
        submission.loc[batch["idx"].numpy()] = pred.numpy()
        acc += int(torch.argmax(outputs.logits, dim=1).item() == labels.item())
    print("Accuracy: ", acc / len(test_data))
    
    submission.to_csv('submission.csv', index_label='idx')