import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [299, 299])


def process_path(filepath, label):
    img = tf.io.read_file(filepath)
    img = decode_img(img)
    return img, label


def prepare_for_testing(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    return ds


def generate_csv(folder, label2int, output_name):
    labels = list(label2int)
    df = pd.DataFrame(columns=["filepath", "label"])
    i = 0
    for label in labels:
        print("Reading", os.path.join(folder, label, "*"))
        for filepath in glob.glob(os.path.join(folder, label, "*")):
            df.loc[i] = [filepath, label2int[label]]
            i += 1
    output_file = f"{output_name}.csv"
    print("Saving", output_file)
    df.to_csv(output_file)


def get_predictions(threshold, x_test, m):
    """
    Returns predictions for binary classification given `threshold`
    For instance, if threshold is 0.3, then it'll output 1 (malignant) for that sample if
    the probability of 1 is 30% or more (instead of 50%)
    """
    y_pred = m.predict(x_test)
    if not threshold:
        threshold = 0.5
    result = np.zeros((584,))
    for i in range(584):
        # test melanoma probability
        if y_pred[i][0] >= threshold:
            result[i] = 1
        # else, it's 0 (benign)
    return result


def plot_confusion_matrix(y_test, y_pred, class_names=("benign", "malignant")):
    cmn = confusion_matrix(y_test, y_pred)
    # Normalise
    cmn = cmn.astype('float') / cmn.sum(axis=1)[:, np.newaxis]
    # print it
    print(cmn)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cmn, annot=True, fmt='.2f',
                xticklabels=[f"pred_{c}" for c in class_names],
                yticklabels=[f"true_{c}" for c in class_names],
                cmap="Blues"
                )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # plot the resulting confusion matrix
    plt.show()
