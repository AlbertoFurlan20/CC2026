from datasets import load_dataset
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    # Load the IMDb dataset
    imdb = load_dataset('imdb')

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(imdb["train"], test_size=0.2, random_state=42)

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Preprocess data
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_train_data = train_data.map(preprocess_function, batched=True)
    tokenized_train_data = tokenized_train_data.remove_columns(["text"])
    tokenized_train_data = tokenized_train_data.rename_column("label", "labels")

    tokenized_test_data = test_data.map(preprocess_function, batched=True)
    tokenized_test_data = tokenized_test_data.remove_columns(["text"])
    tokenized_test_data = tokenized_test_data.rename_column("label", "labels")

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

    # Initialize DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train model
    trainer.train(num_train_epochs=3)

    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Test loss: {eval_results['eval_loss']}")
    print(f"Test accuracy: {eval_results['eval_accuracy']}")

    # Make predictions
    predictions = trainer.predict(tokenized_test_data)
    probs = predictions.predictions

    # Convert predictions to DataFrame
    submission = pd.DataFrame({"probability_class_0": probs[:, 0], "probability_class_1": probs[:, 1]})
    submission.to_csv("submission.csv", index=False)

    # Save the model
    model.save_pretrained('distilbert_model.bin')