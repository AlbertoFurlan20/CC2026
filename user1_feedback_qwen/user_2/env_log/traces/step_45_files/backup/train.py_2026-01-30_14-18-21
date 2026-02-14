import pandas as pd
import numpy as np
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

DIMENSIONS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Load training and test datasets
train_df = pd.read_csv('train.csv', header=0, names=['text_id', 'full_text'] + DIMENSIONS, index_col='text_id')
test_df = pd.read_csv('test.csv', header=0, names=['text_id', 'full_text'] + DIMENSIONS, index_col='text_id')

# Preprocess the text data
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['full_text'])

train_sequences = tokenizer.texts_to_sequences(train_df['full_text'])
test_sequences = tokenizer.texts_to_sequences(test_df['full_text'])

max_length = max([len(s) for s in train_sequences])
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Split the dataset into features and labels
X_train = train_padded
y_train = np.array([train_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X_train))])
X_test = test_padded
y_test = np.array([test_df.drop(['full_text'], axis=1).iloc[i] for i in range(len(X_test))])

# Normalize the labels
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_length),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(DIMENSIONS), activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train_scaled, epochs=3, batch_size=32, validation_split=0.1)

# Predict on the test set
y_test_pred = model.predict(X_test)
y_test_pred = scaler.inverse_transform(y_test_pred)

# Evaluate the model on the test set
metrics = compute_metrics_for_regression(y_test, y_test_pred)
print(metrics)