import os
import json
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# Número de clases del problema (coincide con la salida de la red)
NUM_CLASSES = 290


def label_map(category_str):
    # Quitar corchetes si vienen en el string
    category_str = category_str.strip().strip('[]')

    ids = [int(x.strip()) for x in category_str.split(',') if x.strip() != '']

    labels = np.zeros(NUM_CLASSES, dtype=int)
    for cid in ids:
        labels[cid - 1] = 1
    return labels


if __name__ == "__main__":
    # Directorio donde están las imágenes (dentro del workspace del task)
    image_dir = "images/"

    # Cargar CSV multilabel
    train_df = pd.read_csv("multilabel_classification/train.csv")
    train_df["categories"] = train_df["categories"].apply(label_map)

    # Construir ruta de fichero para cada id
    file_name = []
    for idx in range(len(train_df)):
        file_name.append(image_dir + train_df["id"][idx] + ".png")
    train_df["file_name"] = file_name

    # Cargar imágenes en memoria (simple pero suficiente para este benchmark)
    X_dataset = []
    SIZE = 224

    for i in range(len(train_df)):
        img = load_img(train_df["file_name"][i], target_size=(SIZE, SIZE))
        img = img_to_array(img)
        img = img / 255.0
        X_dataset.append(img)

    X = np.array(X_dataset)
    y = np.array([label_map(categories) for categories in train_df["categories"].tolist()])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=20, test_size=0.3
    )

    # Definir modelo CNN simple
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    # Compilar el modelo
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    # Entrenamiento
    EPOCHS = 10
    BATCH_SIZE = 64

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
    )

    # Generar predicciones para el conjunto de evaluación y guardarlas en submission.csv
    valid_json = json.load(open("object_detection/eval.json"))["images"]
    valid_df = pd.DataFrame(valid_json)

    predict_list = []
    classes = np.array(pd.read_csv("category_key.csv")["name"].to_list())

    for i in range(len(valid_df)):
        img_path = image_dir + valid_df["file_name"][i]
        img = load_img(img_path, target_size=(SIZE, SIZE))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        proba = model.predict(img, verbose=0)[0]  # vector de long NUM_CLASSES

        threshold = 0.5
        predict = []
        for j in range(len(proba)):
            if proba[j] >= threshold:
                predict.append(j + 1)
        predict.sort()
        predict_list.append(predict)

    valid_id = [x[:-4] for x in valid_df["file_name"].to_list()]

    submission