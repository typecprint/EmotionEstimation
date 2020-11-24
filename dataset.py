import matplotlib.pyplot as plt
import tensorflow as tf


def data_augmentation():
    _data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(
                0.02, fill_mode="reflect"
            ),
        ]
    )
    return _data_augmentation


def prepare(
    ds,
    batch_size,
    shuffle_buffer_size,
    shuffle=False,
    augment=None,
    show_sample=True,
):

    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)

    # Batch all datasets
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set
    if augment is not None:
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def create_dataset(
    x,
    y,
    train=False,
    augmentation=False,
    shuffle_buffer_size=100,
    batch_size=64,
    show_augment_sample=False,
    show_sample=False,
):
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if augmentation:
        augment = data_augmentation()
    else:
        augment = None

    if show_augment_sample:
        if augmentation is not True:
            raise ValueError("augmentation must be true.")

        plt.figure(figsize=(10, 10))
        for i in range(16):
            augmented_image = augment(x[:1])
            plt.subplot(4, 4, i + 1)
            plt.imshow(augmented_image[0], cmap="gray")
            plt.axis("off")
        plt.suptitle("augment images.")
        plt.tight_layout()
        plt.show()

    if show_sample:
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(x[i + 1], cmap="gray")
            plt.axis("off")
        plt.suptitle("first 16 samples.")
        plt.tight_layout()
        plt.show()

    if train:
        ds = prepare(
            ds,
            shuffle=True,
            augment=augment,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
        )
    else:
        ds = prepare(ds, batch_size=batch_size, shuffle_buffer_size=shuffle_buffer_size)

    return ds
