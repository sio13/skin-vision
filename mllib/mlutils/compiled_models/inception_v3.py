import tensorflow as tf
import tensorflow_hub as hub

batch_size = 64
optimizer = "rmsprop"

module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
m = tf.keras.Sequential([
    hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
m.build([None, 299, 299, 3])
m.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
m.load_weights("../pretrained/benign-vs-malignant_64_rmsprop_0.442.h5")
