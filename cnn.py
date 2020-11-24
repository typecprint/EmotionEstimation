import tensorflow as tf


def model_CNN(input_shape, output_shape):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(10, 3, input_shape=input_shape, activation="relu"),
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_shape, activation="relu"),
            tf.keras.layers.Softmax(),
        ]
    )
    return model
