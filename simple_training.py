# %%
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# %%
# Load FER2013 Dataset
dataset_path = "/Users/yohei/workspace/dataset/FER2013"
df_all = pd.read_csv(os.path.join(dataset_path, "icml_face_data.csv"))

df_train = df_all[df_all[" Usage"] == "Training"]
df_val = df_all[df_all[" Usage"] == "PrivateTest"]
df_test = df_all[df_all[" Usage"] == "PublicTest"]


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


emotion_filter = [0, 3, 6]
x_train, y_train = arrange_data(df_train, emotion_filter=emotion_filter)
x_val, y_val = arrange_data(df_val, emotion_filter=emotion_filter)
x_test, y_test = arrange_data(df_test, emotion_filter=emotion_filter)

h = v = 48
input_shape = (h, v, 1)
output_shape = len(emotion_filter)

emotion_index_map = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "calm",
}

# %%
# Load MNIST Dataset
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

h = v = 28
input_shape = (h, v, 1)
output_shape = 10


# %%
# Define Convolutional Model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(10, 4, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Conv2D(10, 4, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(10, 4, activation="relu"),
        tf.keras.layers.Conv2D(10, 4, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_shape, activation="relu"),
        tf.keras.layers.Softmax(),
    ]
)
model.summary()

# %%
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="RMSprop", loss=loss_fn, metrics=["accuracy"])

# %%
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_data=(x_val, y_val),
    shuffle=True,
)

# %%
model.evaluate(x_test, y_test, verbose=2)

# %%
output_test = model.predict(x_test)
prob_test = tf.nn.softmax(output_test).numpy()
y_pred = prob_test.argmax(axis=1)

# %%
confusion_matrix(y_test, y_pred)

# %%
