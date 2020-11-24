# %%
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Softmax
from tensorflow.keras import Model

from dataset import create_dataset
from load_data import load_MNIST, load_FER2013


BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100

# %%
# load data
if True:
    output_shape = 3
    emotion_filter = [0, 3, 6]
    x_train, y_train, x_val, y_val, x_test, y_test = load_FER2013(emotion_filter)
    test_ds = create_dataset(x_test, y_test, train=False)
else:
    output_shape = 10
    x_train, y_train, x_val, y_val, x_test, y_test = load_MNIST()
    test_ds = create_dataset(x_test, y_test, train=False)

# %%
model = tf.keras.models.load_model("./model/emopy_3emotion_best")

# %%
# loss_object = tf.keras.losses.CategoricalCrossentropy()
loss_object = tf.keras.losses.CosineSimilarity()

# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.RMSprop()

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


# %%
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

    # print(predictions)
    return predictions


# %%
# evaluation

print("# simple evaluation #")
tamplate = "Test Loss: %.5f, Test Accuracy: %.2f"

y_pred = []
for test_images, test_labels in test_ds:
    _pred = test_step(test_images, test_labels)
    y_pred.extend(_pred.numpy())
y_pred = np.array(y_pred)

print(
    tamplate
    % (
        test_loss.result(),
        test_accuracy.result() * 100,
    ),
    end="\n\n",
)

test_loss.reset_states()
test_accuracy.reset_states()

# %%
y_pred = model.predict(x_test)
print("# confusion matrix #")
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)), end="\n\n")

cr = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("# result summary#")
print(cr, end="\n\n")
# %%
