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

# Load pre-trained GloVe embeddings
embeddings_index = {}
with open('glove.6B.50d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 50))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_length, weights=[embedding_matrix], trainable=False),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(DIMENSIONS), activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train_scaled, epochs=10, batch_size=32, validation_split=0.1)

# Predict on the test set
y_test_pred = model.predict(X_test)
y_test_pred = scaler.inverse_transform(y_test_pred)

# Evaluate the model on the test set
metrics = compute_metrics_for_regression(y_test, y_test_pred)
print(metrics)