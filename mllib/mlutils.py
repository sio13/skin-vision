import tensorflow as tf
import tensorflow_hub as hub


class CheckMySkinModel:
    MODULE_URL = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

    def __init__(self, loss_func="binary_crossentropy", optimizer="rmsprop", model_path=None,
                 weights_path="mllib/pretrained/benign-vs-malignant_64_rmsprop_0.383.h5", sizes=(None, 299, 299, 3)):
        if model_path is None:
            self.model = tf.keras.Sequential([
                hub.KerasLayer(CheckMySkinModel.MODULE_URL, output_shape=[2048], trainable=False),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])
            self.model.build(sizes)
            self.model.compile(loss=loss_func, optimizer=optimizer, metrics=["accuracy"])
            self.model.load_weights(weights_path)

    @staticmethod
    def decode_img(img, sizes=(299, 299)):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, sizes)

    def process_path(self, filepath, label):
        img = tf.io.read_file(filepath)
        img = self.decode_img(img)
        return img, label

    def predict(self, path_to_image, size=(1, 299, 299, 3)):
        test_image = self.process_path(path_to_image, None)
        l, y = test_image
        return self.model.predict(tf.reshape(l, size))


my_model = CheckMySkinModel()
# print(my_model.predict("sample_data/malignant/ISIC_0274382.png"))
