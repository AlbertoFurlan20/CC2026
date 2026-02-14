import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceRegression, DataCollatorWithPadding

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_and_preprocess_data(train_path, test_path):
    # Load training data
    train_df = pd.read_csv(train_path, header=0, names=['text_id', 'full_text', 'Cohesion', 'Syntax', 
                                                        'Vocabulary', 'Phraseology', 'Grammar', 'Conventions'], 
                           index_col='text_id')
    train_df = train_df.dropna(axis=0)

    # Load test data
    test_df = pd.read_csv(test_path, header=0, names=['text_id', 'full_text'], index_col='text_id')

    # Process data and store into numpy arrays.
    X_train = list(train_df.full_text.to_numpy())
    y_train = np.array([train_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X_train))])

    X_test = list(test_df.full_text.to_numpy())

    return X_train, y_train, X_test


def initialize_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceRegression.from_pretrained('bert-base-uncased', num_labels=len(DIMENSIONS))
    return tokenizer, model


if __name__ == '__main__':
    # Define paths
    train_path = 'train.csv'
    test_path = 'test.csv'

    # Load and preprocess data
    X_train, y_train, X_test = load_and_preprocess_data(train_path, test_path)

    # Initialize model
    tokenizer, model = initialize_model()

    # Split training data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=SEED)

    # Define and train the model (structure will be completed later)
    # model = train_model(X_train, y_train, X_valid, y_valid)

    # Evaluate the model on the valid set (structure will be completed later)
    # y_valid_pred = predict(model, X_valid)
    # metrics = compute_metrics_for_regression(y_valid, y_valid_pred)
    # print(metrics)
    # print("final MCRMSE on validation set: ", np.mean(list(metrics.values())))

    # Save submission.csv file for the test set (structure will be completed later)
    # y_submission = predict(model, X_test)
    # submission_df = pd.DataFrame(y_submission, columns=DIMENSIONS)
    # submission_df.index = submission_df.index.rename('text_id')
    # submission_df.to_csv('submission.csv')