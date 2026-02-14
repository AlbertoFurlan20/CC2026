from datasets import load_dataset
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
from transformers import AdamW
from torch.nn import CrossEntropyLoss

if __name__ == "__main__":
    
    # IMPORTANT: Do NOT change this dataset name.
    # The correct HF hub path is "stanfordnlp/imdb".
    imdb = load_dataset("stanfordnlp/imdb")

    # Define the tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

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

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(AdamW(model.parameters(), lr=5e-5), None)  # Use AdamW optimizer
    )

    # Train the model
    trainer.train()

    # Evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(test_data)))
    acc = 0
    for batch in test_dataset:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.softmax(outputs.logits, dim=1)
        submission.loc[batch["idx"].numpy()] = pred.numpy()
        acc += int(torch.argmax(outputs.logits, dim=1).item() == labels.item())
    print("Accuracy: ", acc / len(test_data))
    
    submission.to_csv('submission.csv', index_label='idx')