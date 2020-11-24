# %%
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from load_data import load_FER2013
from cnn import model_CNN
from dataset import create_dataset

EMOTION_FILTER = [0, 3, 6]

h = v = 48
input_shape = (h, v, 1)
output_shape = len(EMOTION_FILTER)

# %%
# load data
x_train, y_train, x_val, y_val, x_test, y_test = load_FER2013(EMOTION_FILTER)

# %%
# create dataset
test_ds = create_dataset(x_test, y_test, train=False, show_sample=True)

# %%
model = model_CNN(input_shape, output_shape)
model.load_weights("./checkpoints/emopy_3emotion")
loss_fn = tf.keras.losses.CosineSimilarity()
model.compile(optimizer="RMSprop", loss=loss_fn, metrics=["accuracy"])
model.summary()

# %%
# evaluation
print("# simple evaluation #")
print(model.evaluate(x_test, y_test, verbose=2), end="\n\n")

y_pred = model.predict(x_test)
print("# confusion matrix #")
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)), end="\n\n")

cr = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("# result summary#")
print(cr, end="\n\n")
# %%
