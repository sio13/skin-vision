import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


def show_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    plt.imshow(np.array(image).reshape(224, 224, 3))
    plt.show()


def get_files(input_folder):
    return list(map(lambda x: os.path.join(input_folder, x), os.listdir(input_folder)))[:10000]


def generate_dataset(input_folder, output_folder):
    benign = get_files(os.path.join(input_folder, 'benign'))
    malignant = get_files(os.path.join(input_folder, 'malignant'))

    for image_path in benign + malignant:
        image_name = image_path.replace(
            input_folder, "").replace(".png", "").replace("/", "").replace("benign", "").replace("malignant", "")

        image = np.array(tf.keras.preprocessing.image.load_img(image_path))
        im_half_length = int(len(image) / 2)
        im1 = Image.fromarray(np.array(list(map(lambda x: x[:im_half_length], image[:im_half_length]))))
        im1.save(os.path.join(output_folder, f"{image_name}_1.png"))

        im2 = Image.fromarray(np.array(list(map(lambda x: x[im_half_length:], image[im_half_length:]))))
        im2.save(os.path.join(output_folder, f"{image_name}_2.png"))

        im3 = Image.fromarray(np.array(list(map(lambda x: x[im_half_length:], image[:im_half_length]))))
        im3.save(os.path.join(output_folder, f"{image_name}_3.png"))

        im4 = Image.fromarray(np.array(list(map(lambda x: x[:im_half_length], image[im_half_length:]))))
        im4.save(os.path.join(output_folder, f"{image_name}_4.png"))


generate_dataset("../../organized_data", "../../enhancer_dataset/raw")
