import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

# Usage:
# predictor = Predictor(CFG, model_path="checkpoints/checkpoint_4.h5")
# print(predictor.predict("../../organized_data/benign/ISIC_0074542.png"))
