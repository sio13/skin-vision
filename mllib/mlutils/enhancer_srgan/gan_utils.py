import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, PReLU
from tensorflow.python.keras.models import Model


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate_psnr(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


def normalize_zero_one_scale(x):
    return x / 255.0


def normalize_minus_one(x):
    return x / 127.5 - 1


def inverse_normalize_minus_one(x):
    return (x + 1) * 127.5


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def res_block(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)
