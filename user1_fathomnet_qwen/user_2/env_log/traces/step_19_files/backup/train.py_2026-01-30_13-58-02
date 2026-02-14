import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
import json

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
    SIZE = 256

    for i in range(len(train_df)):
        img = keras.utils.load_img(train_df["file_name"][i], target_size=(SIZE, SIZE, 3))
        img = keras.utils.img_to_array(img)
        img = img / 255.0
        X_dataset.append(img)

    X = np.array(X_dataset)
    y = np.array(train_df["categories"].to_list())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=20, test_size=0.3
    )

    # Definir modelo CNN simple
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(SIZE, SIZE, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    # Entrenamiento
    EPOCH = 10
    BATCH_SIZE = 64

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train,
        y_train,
        epochs=EPOCH,
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
        img = keras.utils.load_img(img_path, target_size=(SIZE, SIZE, 3))
        img = keras.utils.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        proba = model.predict(img, verbose=0)[0]  # vector de long NUM_CLASSES

        threshold = 0.5
        predict = []
        for j