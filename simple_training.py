# %%
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
# Load FER2013 Dataset
dataset_path = "/Users/yohei/workspace/dataset/FER2013"
train = pd.read_csv(os.path.join(dataset_path, "train.csv"))
img = [np.array(x.split(), dtype=np.int8).reshape(48, 48) for x in train["pixels"]]
label = train["emotion"].values

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

# %%
# Define Convolutional Model
h = v = 28
input_shape = (h, v, 1)
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(10, 4, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Conv2D(10, 4, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.summary()

# %%
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# %%
model.fit(x_train, y_train, epochs=5, batch_size=128)

# %%
model.evaluate(x_test, y_test, verbose=2)

# %%
output_test = model.predict(x_test)
prob_test = tf.nn.softmax(output_test).numpy()
y_pred = prob_test.argmax(axis=1)

# %%
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

# %%
