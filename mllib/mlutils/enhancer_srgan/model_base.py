import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, \
    Lambda
from tensorflow.python.keras.models import Model


class SRGAN:

    def __init__(self, path=None):
        self.model = self.sr_resnet()
        if path:
            self.model.load_weights(path)

    LR_SIZE = 24
    HR_SIZE = 96

    @staticmethod
    def normalize_zero_one_scale(x):
        return x / 255.0

    @staticmethod
    def normalize_minus_one(x):
        return x / 127.5 - 1

    @staticmethod
    def inverse_normalize_minus_one(x):
        return (x + 1) * 127.5

    @staticmethod
    def pixel_shuffle(scale):
        return lambda x: tf.nn.depth_to_space(x, scale)

    def up_sample(self, x_in, num_filters):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = Lambda(self.pixel_shuffle(scale=2))(x)
        return PReLU(shared_axes=[1, 2])(x)

    @staticmethod
    def res_block(x_in, num_filters, momentum=0.8):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = BatchNormalization(momentum=momentum)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Add()([x_in, x])
        return x

    def sr_resnet(self, num_filters=64, num_res_blocks=16):
        x_in = Input(shape=(None, None, 3))
        x = Lambda(self.normalize_zero_one_scale)(x_in)

        x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        for _ in range(num_res_blocks):
            x = self.res_block(x, num_filters)

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x_1, x])

        x = self.up_sample(x, num_filters * 4)
        x = self.up_sample(x, num_filters * 4)

        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        x = Lambda(self.inverse_normalize_minus_one)(x)

        return Model(x_in, x)

    @staticmethod
    def discriminator_block(x_in, num_filters, strides=1, batch_norm=True, momentum=0.8):
        x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
        if batch_norm:
            x = BatchNormalization(momentum=momentum)(x)
        return LeakyReLU(alpha=0.2)(x)

    def discriminator(self, num_filters=64):
        x_in = Input(shape=(SRGAN.HR_SIZE, SRGAN.HR_SIZE, 3))
        x = Lambda(self.normalize_minus_one)(x_in)

        x = self.discriminator_block(x, num_filters, batch_norm=False)
        x = self.discriminator_block(x, num_filters, strides=2)

        x = self.discriminator_block(x, num_filters * 2)
        x = self.discriminator_block(x, num_filters * 2, strides=2)

        x = self.discriminator_block(x, num_filters * 4)
        x = self.discriminator_block(x, num_filters * 4, strides=2)

        x = self.discriminator_block(x, num_filters * 8)
        x = self.discriminator_block(x, num_filters * 8, strides=2)

        x = Flatten()(x)

        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(x_in, x)

    def vgg_22(self):
        return self._vgg(5)

    def vgg_54(self):
        return self._vgg(20)

    def resolve_single(self, model, lr):
        return self.resolve(model, tf.expand_dims(lr, axis=0))[0]

    @staticmethod
    def resolve(model, lr_batch):
        lr_batch = tf.cast(lr_batch, tf.float32)
        sr_batch = model(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch

    @staticmethod
    def _vgg(output_layer):
        vgg = VGG19(input_shape=(None, None, 3), include_top=False)
        return Model(vgg.input, vgg.layers[output_layer].output)

    def enhance(self, image_path, decrease_channels=False):
        image = np.array(Image.open(image_path))
        if decrease_channels:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        sr = self.resolve_single(self.model, image)
        return sr

    @staticmethod
    def compare(lr, sr):
        plt.figure(figsize=(20, 10))

        images = [lr, sr]
        titles = ['Low Resolution', f'Super Resolved (x{sr.shape[0] // lr.shape[0]})']

        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 2, i + 1)
            plt.imshow(img)
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
        plt.show()

