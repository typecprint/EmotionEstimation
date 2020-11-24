import os
import numpy as np
import pandas as pd
import tensorflow as tf


# Load FER2013 Dataset
def arrange_data(_df, emotion_filter=[0, 1, 2, 3, 4, 5, 6]):
    x = np.array(
        [
            np.array(_x.split(), dtype=np.float32).reshape(48, 48)
            for _x in _df[" pixels"]
        ]
    )
    x = x.reshape(-1, 48, 48, 1)
    x /= 255
    y = _df["emotion"].values

    idx = np.array([i for i, _y in enumerate(y) if _y in emotion_filter])
    x = x[idx]
    y = y[idx]
    for i, e in enumerate(emotion_filter):
        y[y == e] = i
    y = tf.keras.utils.to_categorical(y)
    return x, y


def load_FER2013(emotion_filter=[0, 3, 6]):
    dataset_path = "/Users/yohei/workspace/dataset/FER2013"
    df_all = pd.read_csv(os.path.join(dataset_path, "icml_face_data.csv"))

    df_train = df_all[df_all[" Usage"] == "Training"]
    df_val = df_all[df_all[" Usage"] == "PrivateTest"]
    df_test = df_all[df_all[" Usage"] == "PublicTest"]

    x_train, y_train = arrange_data(df_train, emotion_filter=emotion_filter)
    x_val, y_val = arrange_data(df_val, emotion_filter=emotion_filter)
    x_test, y_test = arrange_data(df_test, emotion_filter=emotion_filter)

    return x_train, y_train, x_val, y_val, x_test, y_test


# Load MNIST Dataset
def load_MNIST():
    dataset_path = "/Users/yohei/workspace/dataset/MNIST"
    x_train = np.load(os.path.join(dataset_path, "mnist_train_imgs.npy"))
    y_train = np.load(os.path.join(dataset_path, "mnist_train_labels.npy"))
    x_test = np.load(os.path.join(dataset_path, "mnist_test_imgs.npy"))
    y_test = np.load(os.path.join(dataset_path, "mnist_test_labels.npy"))

    x_train = x_train.reshape([-1, 28, 28, 1])
    x_train = np.array(x_train, dtype=np.float32)
    x_train /= 255

    x_test = x_test.reshape([-1, 28, 28, 1])
    x_test = np.array(x_test, dtype=np.float32)
    x_test /= 255

    return x_train, y_train, _, _, x_test, y_test
