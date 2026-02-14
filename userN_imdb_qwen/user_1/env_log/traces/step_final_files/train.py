from datasets import load_dataset
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

if __name__ == "__main__":
    
    # IMPORTANT: Do NOT change this dataset name.
    # The correct HF hub path is "stanfordnlp/imdb".
    imdb = load_dataset("stanfordnlp/imdb")

    # Preprocess data
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    # Define model here
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer
    )

    # Train model
    trainer.train()

    # Evaluate model using Trainer and save predictions to submission.csv
    predictions, _, metrics = trainer.predict(tokenized_imdb["test"])
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)
    submission = pd.DataFrame(probs.numpy(), columns=["class_0", "class_1"], index=range(len(imdb["test"])))
    submission.to_csv('submission.csv', index_label='idx')

    # Print accuracy
    print("Accuracy: ", metrics["eval_accuracy"])