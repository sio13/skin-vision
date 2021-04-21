import os
from shutil import copyfile

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def get_files(input_folder, samples=2000):
    return list(map(lambda x: os.path.join(input_folder, x), os.listdir(input_folder)))[:samples]


def get_lr_and_hr_bicubic(image, dsize_lr=(56, 56), dsize_hr=(224, 224)):
    lr = cv2.resize(image, dsize=dsize_lr, interpolation=cv2.INTER_CUBIC)
    hr_bicubic = cv2.resize(lr, dsize=dsize_hr, interpolation=cv2.INTER_CUBIC)
    return lr, hr_bicubic


def create_benchmark_dataset(dataset_folder, output_folder, malignant_samples=500, benign_samples=1500):
    benign = get_files(os.path.join(dataset_folder, 'benign'), samples=benign_samples)
    malignant = get_files(os.path.join(dataset_folder, 'malignant'), samples=malignant_samples)

    for benign_image_path in benign:
        image = np.array(tf.keras.preprocessing.image.load_img(benign_image_path))
        image_name = benign_image_path.replace(
            dataset_folder, "").replace(".png", "").replace("/", "").replace("benign", "")
        copyfile(benign_image_path, os.path.join(output_folder, "benign", "hr", f"{image_name}.png"))
        lr, hr_bicubic = get_lr_and_hr_bicubic(image)
        lr_png = Image.fromarray(lr)
        lr_png.save(os.path.join(output_folder, "benign", "lr_x4", f"{image_name}.png"))

        hr_bicubic_png = Image.fromarray(hr_bicubic)
        hr_bicubic_png.save(os.path.join(output_folder, "benign", "hr_bicubic_x4", f"{image_name}.png"))

    for malignant_image_path in malignant:
        image = np.array(tf.keras.preprocessing.image.load_img(malignant_image_path))
        image_name = malignant_image_path.replace(
            dataset_folder, "").replace(".png", "").replace("/", "").replace("malignant", "")
        copyfile(malignant_image_path, os.path.join(output_folder, "malignant", "hr", f"{image_name}.png"))
        lr, hr_bicubic = get_lr_and_hr_bicubic(image)
        lr_png = Image.fromarray(lr)
        lr_png.save(os.path.join(output_folder, "malignant", "lr_x4", f"{image_name}.png"))

        hr_bicubic_png = Image.fromarray(hr_bicubic)
        hr_bicubic_png.save(os.path.join(output_folder, "malignant", "hr_bicubic_x4", f"{image_name}.png"))


create_benchmark_dataset("../../organized_data", "../../benchmark_dataset")
