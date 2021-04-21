import os
import shutil

import pandas as pd


def reorganize_data(path_to_dataset, path_to_images, target):
    dataset = pd.read_csv(path_to_dataset)
    malignant = list(dataset[dataset['target'] == 1]['image_name'])
    benign = list(dataset[dataset['target'] == 0]['image_name'])
    for file_name in malignant:
        full_file_name = os.path.join(path_to_images, file_name + ".png")
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(target, 'malignant', file_name + ".png"))
    for file_name in benign:
        full_file_name = os.path.join(path_to_images, file_name + ".png")
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(target, '../organized_data/benign', file_name + ".png"))


reorganize_data("../data/train.csv", "../data/train", "../organized_data")

