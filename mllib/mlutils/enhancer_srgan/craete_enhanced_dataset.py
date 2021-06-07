import os

import numpy as np
from PIL import Image

from model_base import SRGAN


def get_files(input_folder, filenames):
    return list(map(lambda x: os.path.join(input_folder, x), filenames))


def create_enhanced_dataset(input_folder, output_folder):
    filenames_benign = os.listdir(input_folder)
    paths_benign = get_files(input_folder, filenames_benign)


sr_gan = SRGAN(path="checkpoints/srgan/gan_generator.h5")
sr_gan.compare(np.array(Image.open("marek.png")),
               sr_gan.enhance("marek.png", decrease_channels=True))
