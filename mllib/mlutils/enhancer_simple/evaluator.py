import os

import numpy as np
import tensorflow as tf
from PIL import Image

from model_base import get_prepared_model


def psnr(x1, x2):
    return tf.image.psnr(x2, x1, max_val=255)


def mse(im1, im2):
    return tf.keras.losses.MeanSquaredError()(im1, im2)


def get_dataset(size=250):
    benign_dataset = "../../organized_data/benign"
    malignant_dataset = "../../organized_data/malignant"
    benign = list(map(lambda x: f"{benign_dataset}/{x}", os.listdir(benign_dataset)[:size]))
    malignant = list(map(lambda x: f"{malignant_dataset}/{x}", os.listdir(malignant_dataset)[:size]))
    return benign + malignant


def evaluate_model_scale_four(model, test_image_paths):
    psnr_acc = []
    mse_acc = []
    for i, path in zip(range(len(test_image_paths)), test_image_paths):

        original = Image.open(path)
        downsized = original.resize((original.size[0] // 4, original.size[1] // 4), Image.BICUBIC)
        downsized_as_np = np.array(downsized)
        downsized_as_np.resize([1, original.size[0] // 4, original.size[1] // 4, 3])
        super_resolved = model.predict(downsized_as_np)
        # tf.keras.preprocessing.image.save_img('file.png', super_resolved.reshape([224,224,3]))
        psnr_acc.append(
            psnr(np.array(original).reshape([224, 224, 3]), super_resolved.reshape([224, 224, 3])))  # cv2.PSNR(img1, img2) # tf.image.psnr

        mse_acc.append(mse(np.array(original).reshape([224, 224, 3]), super_resolved.reshape([224, 224, 3])))
        print(psnr(np.array(original).reshape([224, 224, 3]), super_resolved.reshape([224, 224, 3])))
        print(mse(np.array(original).reshape([224, 224, 3]), super_resolved.reshape([224, 224, 3])))

        if i % 30 == 0:
            print(f"{i}-th iteration...")
            tf.keras.preprocessing.image.save_img(f'samples/output_{i}.png', super_resolved.reshape([224, 224, 3]))
            tf.keras.preprocessing.image.save_img(f'samples/input_{i}.png', downsized_as_np.reshape([56, 56, 3]))
            tf.keras.preprocessing.image.save_img(f'samples/original_{i}.png', np.array(original))
    return tf.reduce_mean(psnr_acc), tf.reduce_mean(mse_acc)


if __name__ == '__main__':
    model = get_prepared_model()
    model.load_weights("checkpoints/checkpoint_0.001x.h5")
    test_dataset = get_dataset()

    print(evaluate_model_scale_four(model, test_dataset))
