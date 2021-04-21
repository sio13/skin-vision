import os
import random

random.seed(a=42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import efficientnet.tfkeras as efn
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


def get_model(cfg):
    model_input = tf.keras.Input(shape=(cfg['net_size'], cfg['net_size'], 3), name='imgIn')

    dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

    constructor = getattr(efn, f'EfficientNetB0')
    x = constructor(include_top=False, weights='imagenet',
                    input_shape=(cfg['net_size'], cfg['net_size'], 3),
                    pooling='avg')(dummy)
    x = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)(x)

    result_model = tf.keras.Model(model_input, x)
    return result_model
