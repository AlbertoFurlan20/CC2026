from datasets import load_dataset
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

if __name__ == "__main__":
    
    # Load the IMDb dataset
    imdb = load_dataset("imdb")

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Preprocess data
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    tokenized_imdb = tokenized_imdb.remove_columns(["text"])
    tokenized_imdb = tokenized_imdb.rename_column("label", "labels")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        save_total_limit=1,
        save_strategy="epoch",
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
    )

    # Train model
    for epoch in range(training_args.num_train_epochs):
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Epoch {epoch+1} - Test loss: {eval_results['eval_loss']}")
        print(f"Epoch {epoch+1} - Test accuracy: {eval_results['eval_accuracy']}")

    # Save the model
    model.save_pretrained('distilbert_model.bin')