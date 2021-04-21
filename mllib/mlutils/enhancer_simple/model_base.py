import PIL
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array


def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )


def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    inp = np.expand_dims(y, axis=0)
    out = model.predict(inp)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img


def get_model(upscale_factor=4, channels=3):
    inputs = tf.keras.Input(shape=(56, 56, 3))
    x = tf.keras.layers.Conv2D(64, 5, activation='relu', kernel_initializer='Orthogonal', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='Orthogonal', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='Orthogonal', padding='same')(x)
    x = tf.keras.layers.Conv2D(channels * (upscale_factor ** 2), 3, activation='relu', kernel_initializer='Orthogonal', padding='same')(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return tf.keras.Model(inputs, outputs)


def get_prepared_model():
    model = get_model(upscale_factor=4, channels=3)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


def train_model(model, train_ds, len_df,  batch_size=8, epochs=60):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/checkpoint_{loss:.3f}.h5",
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]
    model.fit(train_ds, steps_per_epoch=len_df // batch_size, epochs=epochs, callbacks=callbacks, verbose=2)  # add validation dataset
    return model
