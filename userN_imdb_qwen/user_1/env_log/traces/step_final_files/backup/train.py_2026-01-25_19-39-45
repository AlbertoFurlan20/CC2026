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

    # Evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(imdb["test"])))
    acc = 0
    for idx, data in enumerate(tokenized_imdb["test"]):
        inputs = tokenizer(data["text"], return_tensors="pt")
        labels = torch.tensor([data["label"]]).unsqueeze(0)
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.softmax(logits, dim=1)
        submission.loc[idx] = pred[0].tolist()
        acc += int(torch.argmax(logits, dim=1).item() == labels.item())
    print("Accuracy: ", acc / len(imdb["test"]))

    submission.to_csv('submission.csv', index_label='idx')