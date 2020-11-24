# %%
from tqdm import tqdm
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
    train_ds = create_dataset(
        x_train,
        y_train,
        train=True,
        show_sample=True,
        show_augment_sample=True,
        batch_size=256,
        augmentation=True,
    )
    val_ds = create_dataset(x_val, y_val, train=False)
    test_ds = create_dataset(x_test, y_test, train=False)
else:
    output_shape = 10
    x_train, y_train, x_val, y_val, x_test, y_test = load_MNIST()
    train_ds = create_dataset(
        x_train,
        y_train,
        train=True,
        show_sample=True,
        batch_size=256,
    )
    test_ds = create_dataset(x_test, y_test, train=False)


# %%
class CNN(Model):
    def __init__(self, output_shape):
        super(CNN, self).__init__()
        self.conv1_1 = Conv2D(10, 3, activation="relu")
        self.conv1_2 = Conv2D(10, 3, activation="relu")
        self.maxpool1 = MaxPool2D(2)
        self.conv2_1 = Conv2D(10, 3, activation="relu")
        self.conv2_2 = Conv2D(10, 3, activation="relu")
        self.maxpool2 = MaxPool2D(2)
        self.flatten = Flatten()
        self.d = Dense(output_shape, activation="relu")
        self.softmax = Softmax()

    def call(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.d(x)
        x = self.softmax(x)
        return x


model = CNN(output_shape)

# %%
# %%
# loss_object = tf.keras.losses.CategoricalCrossentropy()
loss_object = tf.keras.losses.CosineSimilarity()

# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.RMSprop()


train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")


# %%
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# %%
EPOCHS = 50

template_batch = "\rEp:%d Batch:%d/%d loss:%.3f, acc:%.2f"
template_epoch = (
    "\nEpoch:%d, Loss: %.5f, Accuracy: %.2f, Test Loss: %.5f, Test Accuracy: %.2f"
)

min_loss = 1e6
for epoch in range(EPOCHS):
    for n_batch, (images, labels) in enumerate(train_ds):
        train_step(images, labels)
        print(
            template_batch
            % (
                epoch + 1,
                n_batch + 1,
                len(train_ds),
                train_loss.result(),
                train_accuracy.result() * 100,
            ),
            end="",
        )

    for test_images, test_labels in val_ds:
        test_step(test_images, test_labels)

    print(
        template_epoch
        % (
            epoch + 1,
            train_loss.result(),
            train_accuracy.result() * 100,
            test_loss.result(),
            test_accuracy.result() * 100,
        ),
        end="\n",
    )

    if min_loss > test_loss.result():
        print("------------> saving the model.")
        model.save_weights("./checkpoints/emopy_3emotion_best")
        model.save("./model/emopy_3emotion_best")
        min_loss = test_loss.result()

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    print("")

# %%
