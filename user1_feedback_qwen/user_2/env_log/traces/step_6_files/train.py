import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import random

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def compute_metrics_for_regression(y_test, y_test_pred):
    metrics = {}
    for task in DIMENSIONS:
        targets_task = [t[DIMENSIONS.index(task)] for t in y_test]
        pred_task = [l[DIMENSIONS.index(task)] for l in y_test_pred]
        
        rmse = mean_squared_error(targets_task, pred_task, squared=False)

        metrics[f"rmse_{task}"] = rmse
    
    return metrics

def train_model(X_train, y_train, X_valid, y_valid):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMultiLabelSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(DIMENSIONS))

    inputs_train = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
    inputs_valid = tokenizer(X_valid, padding=True, truncation=True, return_tensors="pt")

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_valid_scaled = scaler.transform(y_valid)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(3):  # number of epochs
        optimizer.zero_grad()
        outputs = model(**inputs_train, labels=torch.tensor(y_train_scaled).float())
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs_valid)
        y_valid_pred = outputs.logits.detach().numpy()
        y_valid_pred = scaler.inverse_transform(y_valid_pred)

    return model, y_valid_pred

def predict(model, X):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(X, padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        y_pred = outputs.logits.detach().numpy()
        y_pred = scaler.inverse_transform(y_pred)

    return y_pred

if __name__ == '__main__':
    # Load training and validation datasets
    train_df = pd.read_csv('train.csv', header=0, names=['text_id', 'full_text'] + DIMENSIONS, index_col='text_id')
    train_df = train_df.dropna(axis=0)

    # Split the dataset into features and labels
    X_train = list(train_df.full_text.to_numpy())
    y_train = np.array([train_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X_train))])

    # Create a train-valid split of the data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=SEED)

    # Train the model
    model, y_valid_pred = train_model(X_train, y_train, X_valid, y_valid)

    # Evaluate the model on the valid set using compute_metrics_for_regression and print the results
    metrics = compute_metrics_for_regression(y_valid, y_valid_pred)
    print(metrics)
    print("final M