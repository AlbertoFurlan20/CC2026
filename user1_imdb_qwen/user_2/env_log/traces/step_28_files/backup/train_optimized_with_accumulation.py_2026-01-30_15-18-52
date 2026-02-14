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

    # Convert train_data, val_data, and test_data into dictionaries with 'text' as the key
    train_data = [{"text": data["text"], "label": data["label"]} for data in train_data]
    val_data = [{"text": data["text"], "label": data["label"]} for data in val_data]
    test_data = [{"text": data["text"], "label": data["label"]} for data in test_data]

    # Prepare the data for training by tokenizing the text and creating DataLoader objects
    def tokenize_function(examples):
        return tokenizer([example["text"] for example in examples], padding="max_length", truncation=True)

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
        per_device_train_batch_size=4,  # Reduced from 8 to 4
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",  # Update evaluation strategy to 'steps'
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

    # Train the model with gradient accumulation
    def compute_loss(model, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        return loss

    def train_step(trainer, batch):
        loss = compute_loss(trainer.model, batch)
        loss.backward()
        return loss.item()

    def accumulate_gradients(trainer, dataloader, num_accumulation_steps):
        total_loss = 0.0
        for i, batch in enumerate(dataloader):
            loss = train_step(trainer, batch)
            total_loss += loss
            if (i + 1) % num_accumulation_steps == 0 or i == len(dataloader) - 1:
                trainer.optimizer.step()
                trainer.model.zero_grad()
        return total_loss / len(dataloader)

    # Train the model
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    for epoch in range(training_args.num_train_epochs):
        total_loss = accumulate_gradients(trainer,