import os

import cv2
import numpy as np
from PIL import Image

from model_base import SRGAN


def get_files(input_folder, filenames):
    return list(map(lambda x: os.path.join(input_folder, x), filenames))


def create_enhanced_dataset(input_folder, output_folder):
    filenames_benign = os.listdir(input_folder("benign"))
    paths_benign = get_files(input_folder("benign"), filenames_benign)
    filenames_malignant = os.listdir(input_folder("malignant"))
    paths_malignant = get_files(input_folder("malignant"), filenames_malignant)

    sr_gan = SRGAN(path="checkpoints/srgan/gan_generator.h5")

    for path, filename in zip(paths_benign, filenames_benign):
        new = np.array(sr_gan.enhance(path))
        sr = cv2.resize(new, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        sr_png = Image.fromarray(sr)
        sr_png.save(os.path.join(output_folder("benign"), filename))

    for path, filename in zip(paths_malignant, filenames_malignant):
        new = np.array(sr_gan.enhance(path))
        sr = cv2.resize(new, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        sr_png = Image.fromarray(sr)
        sr_png.save(os.path.join(output_folder("malignant"), filename))


# sr_gan = SRGAN(path="checkpoints/srgan/gan_generator.h5")
# sr_gan.compare(np.array(Image.open("marek.png")),
#                sr_gan.enhance("marek.png", decrease_channels=True))

input_folder = lambda x: f"../../benchmark_dataset/{x}/lr_x4"
output_folder = lambda x: f"../../benchmark_dataset/{x}/sr_x4"
create_enhanced_dataset(input_folder, output_folder)
