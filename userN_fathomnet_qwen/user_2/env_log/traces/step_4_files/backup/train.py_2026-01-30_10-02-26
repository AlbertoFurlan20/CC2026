import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
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

    # Compilar el modelo con una función de pérdida adecuada para multi-label
    def binary_focal_loss(gamma=2., alpha=.25):
        """
        Binary form of focal loss.
           FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
               where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0
        References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
        Usage:
           model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        def binary_focal_loss_fixed(y_true, y_pred):
            """
            :param y_true: A tensor of the same shape as `y_pred`
            :param y_pred:  A tensor resulting from a sigmoid
            :return: Output tensor.
            """
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true