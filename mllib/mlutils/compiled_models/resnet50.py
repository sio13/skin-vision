import tensorflow as tf

batch_size = 64
optimizer = "rmsprop"

base_model = tf.keras.applications.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None
)

m = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(1, activation="sigmoid")
])
m.build([None, 224, 224, 3])
m.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
