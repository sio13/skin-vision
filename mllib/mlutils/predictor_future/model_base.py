import os

import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

CFG = dict(
    batch_size=32,
    read_size=256,
    crop_size=235,
    net_size=224,
    LR=1e-4,
    epochs=30,
    rot=180.0,
    shr=2.0,
    hzoom=8.0,
    wzoom=8.0,
    hshift=8.0,
    wshift=8.0,
    tta_steps=15,
    es_patience=4,
)


class Predictor:

    def __init__(self, cfg, model_path="checkpoints/checkpoint_4.h5"):
        self.cfg = cfg
        self.model = self.get_model()
        self.loaded_model = self.load_model(self.model, model_path)

    @staticmethod
    def plot_confusion_matrix(real, pred):
        print(real)
        print(pred)
        cmn = confusion_matrix(real, pred)
        cmn = cmn.astype('float') / cmn.sum(axis=1)[:, np.newaxis]
        print(cmn)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cmn, annot=True, fmt='.2f',
                    xticklabels=["pred_benign", "pred_malignant"],
                    yticklabels=["true_benign", "true_malignant"],
                    cmap="Blues"
                    )
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        # plot the resulting confusion matrix
        plt.show()

    def get_model(self):
        model_input = tf.keras.Input(shape=(self.cfg['net_size'], self.cfg['net_size'], 3), name='imgIn')

        dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

        constructor = getattr(efn, f'EfficientNetB0')
        x = constructor(include_top=False, weights='imagenet',
                        input_shape=(self.cfg['net_size'], self.cfg['net_size'], 3),
                        pooling='avg')(dummy)
        x = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)(x)

        result_model = tf.keras.Model(model_input, x)
        return result_model

    def prepare(self, img):
        img = tf.image.resize(img, [self.cfg['read_size'], self.cfg['read_size']])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.central_crop(img, self.cfg['crop_size'] / self.cfg['read_size'])
        img = tf.image.resize(img, [self.cfg['net_size'], self.cfg['net_size']])
        img = tf.reshape(img, [self.cfg['net_size'], self.cfg['net_size'], 3])
        return img

    @staticmethod
    def load_model(model_arch, path_h5):
        model_arch.load_weights(path_h5)
        return model_arch

    def show_image(self, image_path):
        image = tf.keras.preprocessing.image.load_img(image_path)
        plt.imshow(self.prepare(np.array(image).reshape(224, 224, 3)))
        plt.show()

    def predict(self, image_path):
        image = tf.keras.preprocessing.image.load_img(image_path)
        prediction = self.loaded_model(np.array([self.prepare(np.array(image).reshape(224, 224, 3))]))
        return float(prediction[0])

    def predict_batch(self, image_list, threshold=0.5):
        images = list(map(tf.keras.preprocessing.image.load_img, image_list))
        predictions = list(
            map(lambda x: self.loaded_model(np.array([self.prepare(np.array(x).reshape(224, 224, 3))]))[0], images))
        return list(map(lambda x: 1 if x > threshold else 0, predictions))

    def evaluate_sens_spec(self, dataset_folder, threshold=0.4, malignant_samples=500, benign_samples=1500):
        benign = list(map(lambda x: os.path.join(dataset_folder, 'benign', x),
                          os.listdir(os.path.join(dataset_folder, 'benign'))))
        malignant = list(map(lambda x: os.path.join(dataset_folder, 'malignant', x),
                             os.listdir(os.path.join(dataset_folder, 'malignant'))))
        real = [0] * len(benign[:benign_samples]) + [1] * len(malignant[:malignant_samples])
        mal = self.predict_batch(malignant[:malignant_samples], threshold)
        ben = self.predict_batch(benign[:benign_samples], threshold)
        predicted = mal + ben

        self.plot_confusion_matrix(real, predicted)

    def evaluate(self, dataset_folder, threshold=0.3, malignant_samples=500, benign_samples=1500):

        benign = list(map(lambda x: os.path.join(dataset_folder, 'benign', x),
                          os.listdir(os.path.join(dataset_folder, 'benign'))))[:2000]
        malignant = list(map(lambda x: os.path.join(dataset_folder, 'malignant', x),
                             os.listdir(os.path.join(dataset_folder, 'malignant'))))

        mal = self.predict_batch(malignant[:malignant_samples], threshold)
        ben = self.predict_batch(benign[:benign_samples], threshold)

        correct = mal.count(1) + ben.count(0)
        return correct / (malignant_samples + benign_samples)


# Usage:
predictor = Predictor(CFG, model_path="checkpoints/checkpoint_4.h5")
# print(predictor.predict("../../organized_data/benign/ISIC_0074542.png"))
# predictor.evaluate_sens_spec("../../organized_data", 0.3)
print(predictor.evaluate("../../organized_data", 0.3))
