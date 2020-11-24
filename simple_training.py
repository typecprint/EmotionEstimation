# %%
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder

from cnn import model_CNN
from load_data import load_FER2013
from dataset import create_dataset


emotion_index_map = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "calm",
}
EMOTION_FILTER = [0, 3, 6]

h = v = 48
input_shape = (h, v, 1)
output_shape = len(EMOTION_FILTER)

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100

# %%
# load data
x_train, y_train, x_val, y_val, x_test, y_test = load_FER2013(EMOTION_FILTER)

# %%
# create dataset
train_ds = create_dataset(x_train, y_train, train=True, show_sample=True)
val_ds = create_dataset(x_val, y_val, train=False)
test_ds = create_dataset(x_test, y_test, train=False)

# %%
# Define Convolutional Model
model = model_CNN(input_shape, output_shape)
model.summary()

# %%
# loss_fn = tf.keras.losses.CategoricalCrossentropy()
loss_fn = tf.keras.losses.CosineSimilarity()
model.compile(optimizer="RMSprop", loss=loss_fn, metrics=["accuracy"])
model.fit(train_ds, epochs=50, validation_data=val_ds)

# %%
model.save_weights("./checkpoints/emopy_3emotion")

# %%
model.save("./model/emopy_3emotion")
model.save("./model/emopy_3emotion/model.h5")

# %%
