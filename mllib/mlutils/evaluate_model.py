import os

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import generate_csv, process_path, prepare_for_testing, get_predictions, plot_confusion_matrix
from compiled_models.inception_v3 import m

batch_size = 64
optimizer = "rmsprop"


def evaluate(compiled_model, data_path, test_samples=584, need_generate=False):
    if need_generate:
        generate_csv(data_path, {"malignant": 0, "benign": 1}, os.path.join("..", "csv_data", "eval_data"))

    df_test = pd.read_csv(f'{os.path.join("..", "csv_data", "eval_data")}.csv')
    test_ds = tf.data.Dataset.from_tensor_slices((df_test["filepath"], df_test["label"]))
    test_ds = test_ds.map(process_path)
    test_ds = prepare_for_testing(test_ds, cache="my-new-test-cached-data")

    y_test = np.zeros((test_samples,))
    x_test = np.zeros((test_samples, 299, 299, 3))
    for i, (img, label) in enumerate(test_ds.take(test_samples)):
        x_test[i] = img
        y_test[i] = label.numpy()

    loss, accuracy = compiled_model.evaluate(x_test, y_test, verbose=0)
    print(loss, accuracy)
    threshold = 0.5
    y_pred = get_predictions(threshold, x_test, compiled_model)
    plot_confusion_matrix(y_test, y_pred)


evaluate(m, "../organized_data/", need_generate=False)
