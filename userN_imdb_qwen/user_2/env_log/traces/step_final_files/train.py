from datasets import load_dataset
import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments

if __name__ == "__main__":
    
    # IMPORTANT: Do NOT change this dataset name.
    # The correct HF hub path is "stanfordnlp/imdb".
    imdb = load_dataset("stanfordnlp/imdb")

    def fine_tune_distilbert(dataset):
        model_name = 'distilbert-base-uncased'
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
        )

        trainer.train()

        return model

    #TODO: preprocess data

    #TODO: define model here
    model = fine_tune_distilbert(imdb)

    #evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(imdb["test"])))
    acc = 0
    for idx, data in enumerate(imdb["test"]):
        text = data["text"]
        label = data["label"]
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.softmax(outputs.logits, dim=1)
        submission.loc[idx] = pred[0].tolist()
        acc += int(torch.argmax(outputs.logits, dim=1).item() == label)
    print("Accuracy: ", acc/len(imdb["test"]))
    
    submission.to_csv('submission.csv', index_label='idx')