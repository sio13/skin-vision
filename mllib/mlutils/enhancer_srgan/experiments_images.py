import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gan_utils import psnr
from model_base import SRGAN

sr_gan = SRGAN(path="checkpoints/srgan/gan_generator.h5")


def compare_more(lr, sr, hr, save=False):
    plt.figure(figsize=(20, 10))

    images = [lr, sr, hr]
    titles = ['Low Resolution', f'Super Resolved (x{sr.shape[0] // lr.shape[0]})', "High resolution"]

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    if save:
        plt.savefig("compare_fig_mel.png")
    else:
        plt.show()


def lr_lenna():
    image_path = "../../sample_data/lenna/img.png"
    image = np.array(Image.open(image_path))
    sr_gan.compare(image, sr_gan.enhance(image_path, decrease_channels=True), save=True)


def hr_lenna():
    image_path = "../../sample_data/lenna/img_hr.png"
    image = np.array(Image.open(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    lr = cv2.resize(image, dsize=(int(image.shape[0] / 4), int(image.shape[1] / 4)), interpolation=cv2.INTER_CUBIC)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGRA2BGR)
    hr_bicubic = cv2.resize(image, dsize=(int(image.shape[0]), int(image.shape[1])), interpolation=cv2.INTER_CUBIC)
    hr_bicubic = cv2.cvtColor(hr_bicubic, cv2.COLOR_BGRA2BGR)
    sr = sr_gan.enhance_image(lr, decrease_channels=True)
    print(lr.shape)
    print(sr.shape)
    print(psnr(hr_bicubic, image))
    print(psnr(sr, image))


def mal_mel():
    image_path = "../../sample_data/malignant/ISIC_0274382.png"
    image = np.array(Image.open(image_path))
    lr = cv2.resize(image, dsize=(int(image.shape[0] / 4), int(image.shape[1] / 4)), interpolation=cv2.INTER_CUBIC)
    sr = sr_gan.enhance_image(lr, decrease_channels=True)
    compare_more(lr, sr, image, save=True)


mal_mel()
