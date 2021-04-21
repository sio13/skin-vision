import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from config import optimizer, shape
from predictor_base import process_path, prepare_for_training, prepare_for_testing, plot_confusion_matrix


def get_model():
    module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
    m = tf.keras.Sequential([
        hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    m.build(shape)
    m.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return m


def load_model(model, path):
    model.load_weights(path)
    return model


def get_predictions(model, x_test, n_testing_samples, threshold=0.5):
    y_pred = model.predict(x_test)
    result = np.zeros((n_testing_samples,))
    for i in range(n_testing_samples):
        # test melanoma probability
        if y_pred[i][0] >= threshold:
            result[i] = 1
        # else, it's 0 (benign)
    return result


def train_model(model, batch_size, train_ds):
    train_ds = train_ds.map(process_path)
    train_ds = prepare_for_training(train_ds, batch_size=batch_size, cache="train-cached-data")
    model_name = f"benign-vs-malignant_{batch_size}_{optimizer}"
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs", model_name))

    # TODO for now without validation
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name + "_{loss:.3f}.h5", save_best_only=True,
                                                          verbose=1)

    model.fit(train_ds,
              steps_per_epoch=len(train_ds) // batch_size,
              verbose=1, epochs=100,
              callbacks=[tensorboard, model_checkpoint])


def eval_model(model, test_ds):
    test_ds = test_ds.map(process_path)
    test_ds = prepare_for_testing(test_ds, cache="test-cached-data")
    y_test = np.zeros((len(test_ds),))
    x_test = np.zeros((len(test_ds), 299, 299, 3))
    for i, (img, label) in enumerate(test_ds.take(len(test_ds))):
        x_test[i] = img
        y_test[i] = label.numpy()

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss:", loss, "  Accuracy:", accuracy)

    y_pred = get_predictions(model, x_test, len(test_ds))
    plot_confusion_matrix(y_test, y_pred)
