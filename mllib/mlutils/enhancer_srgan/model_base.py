import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, \
    Lambda
from tensorflow.python.keras.models import Model

from gan_utils import (evaluate_psnr, vgg_54, normalize_zero_one_scale, normalize_minus_one,
                       inverse_normalize_minus_one, pixel_shuffle, res_block, resolve_single, _vgg)


class TrainerBase:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./checkpoints/srgan'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate_psnr(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')

    @property
    def model(self):
        return self.checkpoint.model

    @model.setter
    def model(self, value):
        self.checkpoint.model = value

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            if step % evaluate_every == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()

                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset)

                duration = time.perf_counter() - self.now
                print(
                    f'{step}/{steps}: loss = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')

                if save_best_only and psnr_value <= ckpt.psnr:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.psnr = psnr_value
                ckpt_mgr.save()

                self.now = time.perf_counter()


class GenPreTrainerBase(TrainerBase):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=False):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class GanTrain:

    def __init__(self,
                 generator,
                 discriminator,
                 content_loss='VGG54',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        self.vgg = vgg_54()
        self.content_loss = content_loss
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss  # defined in the paper
            disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss

    def train(self, train_dataset, steps=100000):
        pls_metric = Mean()
        dls_metric = Mean()
        step = 0

        for lr, hr in train_dataset.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 10 == 0:
                print(
                    f'{step}/{steps}, Adv loss = {pls_metric.result():.4f}, D loss = {dls_metric.result():.4f}')
                pls_metric.reset_states()
                dls_metric.reset_states()


class SRGAN:
    def __init__(self, path=None):
        self.model = self.sr_resnet()
        if path:
            self.model.load_weights(path)

    def up_sample(self, x_in, num_filters):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
        x = Lambda(pixel_shuffle(scale=2))(x)
        return PReLU(shared_axes=[1, 2])(x)

    def sr_resnet(self, num_filters=64, num_res_blocks=16):
        x_in = Input(shape=(None, None, 3))
        x = Lambda(normalize_zero_one_scale)(x_in)

        x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        for _ in range(num_res_blocks):
            x = res_block(x, num_filters)

        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x_1, x])

        x = self.up_sample(x, num_filters * 4)
        x = self.up_sample(x, num_filters * 4)

        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        x = Lambda(inverse_normalize_minus_one)(x)

        return Model(x_in, x)

    @staticmethod
    def discriminator_block(x_in, num_filters, strides=1, batch_norm=True, momentum=0.8):
        x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
        if batch_norm:
            x = BatchNormalization(momentum=momentum)(x)
        return LeakyReLU(alpha=0.2)(x)

    def discriminator(self, num_filters=64):
        x_in = Input(shape=(96, 96, 3))
        x = Lambda(normalize_minus_one)(x_in)

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

    def vgg_54(self):
        return _vgg(20)

    def enhance_image(self, image, decrease_channels=False):
        if decrease_channels:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        sr = resolve_single(self.model, image)
        return sr

    def enhance(self, image_path, decrease_channels=False):
        image = np.array(Image.open(image_path))
        return self.enhance_image(image, decrease_channels)

    @staticmethod
    def bicubic(lr):
        sr = cv2.resize(lr, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        return sr

    @staticmethod
    def compare(lr, sr, save=False):
        plt.figure(figsize=(20, 10))

        images = [lr, sr]
        titles = ['Low Resolution', f'Super Resolved (x{sr.shape[0] // lr.shape[0]})']

        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 2, i + 1)
            plt.imshow(img)
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
        if save:
            plt.savefig("compare_fig.png")
        else:
            plt.show()
