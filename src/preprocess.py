import numpy as np
import tensorflow as tf
import os
import gzip
import struct

from config.config import DATA_RAW_DIR


def load_digit_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)


# 🔥 NO TFDS — PURE FILE LOADER
def load_character_data():
    print("Loading EMNIST manually (NO TFDS)...")

    def read_images(path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows, cols)
        return images

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    base_path = os.path.join(DATA_RAW_DIR, "emnist")

    x_train = read_images(os.path.join(base_path, "emnist-balanced-train-images-idx3-ubyte.gz"))
    y_train = read_labels(os.path.join(base_path, "emnist-balanced-train-labels-idx1-ubyte.gz"))

    x_test = read_images(os.path.join(base_path, "emnist-balanced-test-images-idx3-ubyte.gz"))
    y_test = read_labels(os.path.join(base_path, "emnist-balanced-test-labels-idx1-ubyte.gz"))

    # Fix rotation
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = np.transpose(x_test, (0, 2, 1))

    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)