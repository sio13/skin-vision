import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from model_base import SRGAN, GenPreTrainerBase, GanTrain


def get_hr_dataset(path_to_data):
    data_paths = list(map(lambda x: os.path.join(path_to_data, x), os.listdir(path_to_data)))[:100]
    loaded_data = np.array(list(
        map(lambda x: cv2.resize(np.array(tf.keras.preprocessing.image.load_img(x)), dsize=(128, 128),
                                 interpolation=cv2.INTER_CUBIC), data_paths)))

    return loaded_data


def get_lr_dataset(hr_dataset):
    return np.array(list(
        map(lambda x: cv2.resize(x, dsize=(int(128 / 4), int(128 / 4)),
                                 interpolation=cv2.INTER_CUBIC), hr_dataset)))


def get_dataset(path_to_data, batch_size=16, repeat_count=None):
    hr = get_hr_dataset(path_to_data)
    lr = get_lr_dataset(hr)

    lr_dataset = tf.data.Dataset.from_tensor_slices(lr)
    hr_dataset = tf.data.Dataset.from_tensor_slices(hr)

    ds = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds, ds.take(10)


srgan = SRGAN()
train_ds, valid_ds = get_dataset("../../enhancer_dataset/raw")
print("dataset generated")
pre_trainer = GenPreTrainerBase(model=srgan.sr_resnet(), checkpoint_dir='weights')
print("trainer created")

pre_trainer.train(train_ds, valid_ds.take(10), steps=100000, evaluate_every=1000)

pre_trainer.model.save_weights('weights/pretrained_generator.h5')

print("generator pretrained")

gan_generator = srgan.sr_resnet()
gan_generator.load_weights('weights/srgan/pre_generator.h5')

trained_gan = GanTrain(generator=gan_generator, discriminator=srgan.discriminator())
trained_gan.train(train_ds, steps=200000)

# final results
trained_gan.generator.save_weights('weights/generator.h5')
