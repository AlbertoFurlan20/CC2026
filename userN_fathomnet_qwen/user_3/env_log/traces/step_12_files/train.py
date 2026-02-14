import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
        img = preprocess_input(img)
        X_dataset.append(img)

    X = np.array(X_dataset)
    y = np.array(train_df["categories"].to_list())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=20, test_size=0.3
    )

    # Definir modelo ResNet pre-entrenado
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Congelar capas del ResNet
    for layer in base_model.layers:
        layer.trainable = False

    # Compilar modelo
    model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

    # Entrenamiento
    EPOCH = 1
    BATCH_SIZE = 64

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
        img = preprocess_input(img)
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
    valid_osd = [1] * len(valid_id)

    submit_data = [
        [valid_id[i], predict_list[i], valid_osd[i]] for i in range(len(valid_id))
    ]
    pd.DataFrame(data=submit_data, columns=["id", "categories", "osd"]).to_csv(
        "submission.csv", index=False
    )