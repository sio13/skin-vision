import random

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# to get consistent results after multiple runs
tf.random.set_seed(7)
np.random.seed(7)
random.seed(7)
batch_size = 64
optimizer = "rmsprop"

# 0 for benign, 1 for malignant
class_names = ["benign", "malignant"]


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, (299, 299))


def process_path(filepath, label):
    # load the raw data from the file as a string
    img = tf.io.read_file(filepath)
    img = decode_img(img)
    return img, label


module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
m = tf.keras.Sequential([
    hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

m.build((None, 299, 299, 3))
m.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
m.summary()

m.load_weights("../pretrained/benign-vs-malignant_64_rmsprop_0.383.h5")
# tf.keras.scripts.load_model(model_path)

# 15 % threshold so far


test_image = process_path("../sample_data/malignant/ISIC_0274382.png", 1)
l, y = test_image
print(m.predict(tf.reshape(l, (1, 299, 299, 3))))
# work with threshold about 15% please for now
