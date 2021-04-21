import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from config import class_names


def generate_csv(folder, label2int):
    folder_name = os.path.basename(folder)
    labels = list(label2int)
    # generate CSV file
    df = pd.DataFrame(columns=["filepath", "label"])
    i = 0
    for label in labels:
        print("Reading", os.path.join(folder, label, "*"))
        for filepath in glob.glob(os.path.join(folder, label, "*")):
            df.loc[i] = [filepath, label2int[label]]
            i += 1
    output_file = f"{folder_name}.csv"
    print("Saving", output_file)
    df.to_csv(output_file)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [299, 299])


def process_path(filepath, label):
    # load the raw data from the file as a string
    img = tf.io.read_file(filepath)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, cache=True, batch_size=64, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)

    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def prepare_for_testing(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    return ds


def show_batch(train_ds):
    batch = next(iter(train_ds))
    plt.figure(figsize=(12, 12))
    for n in range(25):
        _ = plt.subplot(5, 5, n + 1)
        plt.imshow(batch[0][n])
        plt.title(class_names[batch[1][n].numpy()].title())
        plt.axis('off')
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
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
