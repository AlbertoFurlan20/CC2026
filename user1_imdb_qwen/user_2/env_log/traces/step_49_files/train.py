import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Import necessary libraries for loading the dataset and using DistilBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load the IMDb dataset using HuggingFace's datasets library
dataset = load_dataset("imdb")

# 3. Split the dataset into training and validation sets using sklearn's train_test_split
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# 4. Initialize the DistilBERT model and set up the training arguments
num_labels = 2  # Set the number of labels for the DistilBERT model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).to(device)

def tokenize_function(examples):
    try:
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    except KeyError as e:
        raise ValueError(f"Missing column: {e}")

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
tokenized_datasets = {'train': tokenized_train_dataset, 'validation': tokenized_val_dataset}

required_columns = ['input_ids', 'attention_mask', 'labels']
for split in ['train', 'validation']:
    if not all(column in tokenized_datasets[split].column_names for column in required_columns):
        raise ValueError(f"Missing required columns in {split} dataset: {set(required_columns) - set(tokenized_datasets[split].column_names)}")

tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['text'])
tokenized_datasets['validation'] = tokenized_datasets['validation'].remove_columns(['text'])
tokenized_datasets['train'] = tokenized_datasets['train'].rename_column('label', 'labels')
tokenized_datasets['validation'] = tokenized_datasets['validation'].rename_column('label', 'labels')
tokenized_datasets['train'].set_format('torch')
tokenized_datasets['validation'].set_format('torch')

train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['validation']

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 5. Fine-tune the model on the training dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# 6. Save the per-class probabilities for the test set examples to submission.csv
predictions = trainer.predict(test_dataset)
probabilities = torch.nn.functional.softmax(predictions.predictions, dim=-1)
submission = pd.DataFrame({
    'id': test_dataset['idx'],
    'probability_pos': probabilities[:, 1]
})
submission.to_csv('submission.csv', index=False)