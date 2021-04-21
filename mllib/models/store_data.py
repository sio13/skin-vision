import os
import zipfile

from tensorflow.keras.utils import get_file


def download_and_extract_dataset():
    train_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip"
    valid_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip"
    test_url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip"
    for i, download_link in enumerate([valid_url, train_url, test_url]):
        temp_file = f"{os.path.join(os.path.join(os.getcwd(), '..', 'net_data'))}/temp{i}.zip"
        data_dir = get_file(origin=download_link, fname=os.path.join(os.path.join(os.getcwd(), "..", "net_data"), temp_file))
        print("Extracting", download_link)
        with zipfile.ZipFile(data_dir, "r") as z:
            z.extractall("data")
        # remove the temp file
        os.remove(temp_file)


# comment the below line if you already downloaded the dataset
download_and_extract_dataset()
