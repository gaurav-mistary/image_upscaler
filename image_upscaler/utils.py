import os
import time
import wget
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
from skimage import data, io, filters
from numpy import array
from image_upscaler.settings import BASE_DIR
from math import ceil

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Declaring Constants
ESARGAN_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
SAVED_MODEL_PATH = BASE_DIR / "gen_model3000.h5"


def get_esargan_model():
  return hub.load(ESARGAN_MODEL_PATH)


def get_model(custom_objects=None):
    if not os.path.exists(SAVED_MODEL_PATH):
        wget.download(
            "https://github.com/deepak112/Keras-SRGAN/raw/master/model/gen_model3000.h5"
        )

    return tf.keras.models.load_model(SAVED_MODEL_PATH, custom_objects=custom_objects)


def preprocess_image(image_path):
    """Loads image from path and preprocesses to make it model ready
    Args:
      image_path: Path to the image file
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)


def save_image(image, filename):
    """
        Saves unscaled Tensor Images.
        Args:plot_image(tf.squeeze(fake_image), "Super Resolution")
    plt.savefig("ESRGAN_DIV2K.jpg", bbox_inches="tight")
    print("PSNR: %f" % psnr)
          image: 3D image tensor. [height, width, channels]
          filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)


# Defining helper functions
def downscale_image(image):
    """
    Scales down images using bicubic downsampling.
    Args:
        image: 3D or 4D tensor of preprocessed image
    """
    image_size = []
    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError("Dimension mismatch. Can work only on single image.")

    image = tf.squeeze(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8))

    lr_image = np.asarray(
        Image.fromarray(image.numpy()).resize(
            [image_size[0] // 4, image_size[1] // 4], Image.BICUBIC
        )
    )

    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image


def plt_save_image(image, filename):
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)


class VGG_LOSS(object):
    def __init__(self, image_shape):

        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):

        vgg19 = VGG19(
            include_top=False, weights="imagenet", input_shape=self.image_shape
        )
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(
            inputs=vgg19.input, outputs=vgg19.get_layer("block5_conv4").output
        )
        model.trainable = False

        return K.mean(K.square(model(y_true) - model(y_pred)))


def get_optimizer():
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam


# work on image
def load_data_from_img(this_img):
    return io.imread(this_img)


def lr_images(images_real):
    images = []
    print(images_real.shape)
    # images.append(cv2.resize(images_real, (128,128), interpolation=cv2.INTER_CUBIC))
    images.append(cv2.resize(images_real, (96, 96), interpolation=cv2.INTER_CUBIC))
    images_lr = array(images)
    return images_lr


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    # input_data = (input_data + 1) * 127.5
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def get_intermediate(image_path):
    x_test_lr = lr_images(load_data_from_img(image_path))
    x_test_lr = normalize(x_test_lr)

    loss = VGG_LOSS((96, 96, 3))
    model = get_model(custom_objects={"vgg_loss": loss.vgg_loss})
    layer_outputs = [layer.output for layer in model.layers[0:111]]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(x_test_lr)
    layer_names = []
    for layer in model.layers[0:111]:
        layer_names.append(layer.name)

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        images_per_row = min(16, n_features)
        n_cols = int(ceil(n_features / images_per_row))
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        # print(layer_name, size,n_features, n_cols)

        for col in range(images_per_row):
            for row in range(n_cols):
                channel_image = layer_activation[0, :, :, row * images_per_row + col]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[
                    row * size : (row + 1) * size, col * size : (col + 1) * size
                ] = channel_image
        scale = 1.0 / size
        plt.figure(
            figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0])
        )
        plt.title(layer_name)
        plt.grid(False)

        # plt.imshow(display_grid, aspect='auto', cmap='viridis')
        ## This below line on uncomment will save images like last time

        static_path = BASE_DIR / "static" / "img" / "intermediate"

        plt.savefig(static_path / f"{layer_name}.jpg", bbox_inches="tight")
    
    return layer_names