# train.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding for the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))  # New layer
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # New output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001)

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val), callbacks=[early_stopping])

# Plot the training and validation accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Define a function to evaluate the model with different feature selections
def evaluate_model(model, x_train, y_train, x_val, y_val, feature_selection):
    # Apply the feature selection method
    if feature_selection == 'pca':
        pca = PCA(n_components=0.95)
        x_train_pca = pca.fit_transform(x_train)
        x_val_pca = pca.transform(x_val)
    elif feature_selection == 'kbest':
        selector = SelectKBest(f_classif, k=100)
        x_train_kbest = selector.fit_transform(x_train, y_train)
        x_val_kbest = selector.transform(x_val)

    # Train the model with the selected features
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

    # Evaluate the model
    loss, accuracy = model.evaluate(x_val, y_val)
    return loss, accuracy

# Define a list of feature selection methods
feature_selection_methods = ['pca', 'kbest']

# Iterate over the feature selection methods and evaluate the model
for feature_selection in feature_selection_methods:
    loss, accuracy = evaluate_model(model, x_train, y_train, x_val, y_val, feature_selection)
    print(f'Feature selection method: {feature_selection}, Loss: {loss}, Accuracy: {accuracy}')

# Define a function to train the model with different architectures
def train_model(model, x_train, y_train, x_val, y_val):
    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

    # Plot the training and validation accuracy and loss
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    # Evaluate the model
    loss, accuracy = model.evaluate(x_val, y_val)
    return loss, accuracy

# Define a list of model architectures
model_architectures = [
    {'name': 'model1', 'layers': [128, 64, 290]},
    {'name': 'model2', 'layers': [256, 128, 512]},
    {'name': 'model3', 'layers': [512, 256, 128]}
]

# Iterate over the model architectures and train the model
for model_architecture in model_architectures:
    model = Sequential()
    for i, layer in enumerate(model_architecture['layers']):
        if i == 0:
            model.add(Flatten())
        elif i == 1:
            model.add(Dense(layer, activation='relu'))
            model.add(Dropout(0.5))
        else:
            model.add(Dense(layer, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    loss, accuracy = train_model(model, x_train, y_train, x_val, y_val)
    print(f'Model architecture: {model_architecture["name"]}, Loss: {loss}, Accuracy: {accuracy}')