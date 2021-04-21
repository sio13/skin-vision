import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

from model_base import get_prepared_model, train_model
import numpy as np
batch_size = 8
crop_size = 224
upscale_factor = 4
input_size = crop_size // upscale_factor
dataset = "../../organized_data"


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [crop_size, crop_size])


def process_path(filepath):
    img = tf.io.read_file(filepath)
    img = decode_img(img)
    return img


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

def process_input(inp):
    return tf.image.resize(inp, [56, 56], method="area")


def process_target(inp):
    inp = tf.image.rgb_to_yuv(inp)
    last_dimension_axis = len(inp.shape) - 1
    y, u, v = tf.split(inp, 3, axis=last_dimension_axis)
    return y

df = pd.read_csv("../../mlutils/predictor_simple/data/organized_data.csv")
len_df = len(df)
df_train, df_test = train_test_split(df, test_size=0.1, random_state=False)
train_ds = tf.data.Dataset.from_tensor_slices((df_train["filepath"]))
train_ds = train_ds.map(process_path)
train_ds = prepare_for_training(train_ds, batch_size=batch_size)


train_ds = train_ds.map(
    lambda x: (process_input(x), x)
)


model = train_model(get_prepared_model(), train_ds, len_df)
