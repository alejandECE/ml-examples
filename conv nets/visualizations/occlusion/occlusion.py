#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
from typing import Tuple
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import datetime
import os

# Constants
DATASETS = pathlib.Path(os.environ['DATASETS'])
OUTPUTS = pathlib.Path(os.environ['OUTPUTS'])
RESNET50 = pathlib.Path(os.environ['WEIGHTS']) / 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
BUFFER_SIZE = BATCH_SIZE * 32
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Loads raw image
def load_raw_image(path: tf.Tensor) -> tf.Tensor:
    serialized = tf.io.read_file(path)
    image = tf.io.decode_image(serialized, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], antialias=True)
    return image


# Preprocess image according to pretrained network
def preprocess_image(img: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.resnet50.preprocess_input(img)


# Gets location of the occlusion are as a tensor of 1D of shape [2,]
def get_occlusion_location(window_size: tf.Tensor, output_size: tf.Tensor, index: tf.Tensor) -> tf.Tensor:
    step = (tf.convert_to_tensor(IMG_HEIGHT, dtype=window_size.dtype) - window_size) / output_size
    row = tf.cast(tf.cast((index // output_size), dtype=step.dtype) * step, dtype=tf.int64)
    step = (tf.convert_to_tensor(IMG_WIDTH, dtype=window_size.dtype) - window_size) / output_size
    col = tf.cast(tf.cast((index % output_size), dtype=step.dtype) * step, dtype=tf.int64)
    return tf.stack((row, col))


# Gets occlusion mask for a given location in the image
def get_occlusion_mask(coordinates: tf.Tensor, window_size: tf.Tensor) -> tf.Tensor:
    # Some initial conversions
    img_height = tf.convert_to_tensor(IMG_HEIGHT, dtype=tf.int64)
    img_width = tf.convert_to_tensor(IMG_WIDTH, dtype=tf.int64)
    # Creates top-surrounding rows
    top_rows = tf.ones(shape=(coordinates[0], img_width))
    # Creates rows with occlusion
    middle_rows = tf.repeat(
        tf.concat(values=(
            tf.ones(shape=(1, coordinates[1])),
            tf.zeros(shape=(1, tf.minimum(img_width - coordinates[1], window_size))),
            tf.ones(shape=(1, tf.maximum(img_width - coordinates[1] - window_size, 0)))
        ), axis=1),
        repeats=[tf.minimum(img_width - coordinates[1], window_size)], axis=0
    )
    # Creates bottom-surrounding rows
    bottom_rows = tf.ones(shape=(tf.maximum(img_height - coordinates[0] - window_size, 0), img_width))
    # Creates mask putting together all previously generated rows
    mask = tf.concat(
        values=(
            top_rows,
            middle_rows,
            bottom_rows
        ),
        axis=0
    )
    return tf.expand_dims(mask, axis=-1)


# Occludes an image at a given location (defined by index!)
def occlude_image(image: tf.Tensor,
                  window_size: tf.Tensor,
                  output_size: tf.Tensor,
                  index: tf.Tensor) -> tf.Tensor:
    # Gets location of the occlusion are as a tensor of 1D of shape [2,]
    coordinates = get_occlusion_location(window_size, output_size, index)
    # Gets the occlusion mask corresponding to index
    mask = get_occlusion_mask(coordinates, window_size)
    return image * tf.cast(mask, dtype=image.dtype)


# Generates a dataset to feed batches of occluded images
def create_occlusion_dataset(image: str, window_size: int, output_size: int) -> Tuple:
    # Loads records from files
    image_ds = tf.data.Dataset.from_tensors(image)
    # Parses image and repeats it as many times as points will be in the heat map
    image_ds = image_ds.map(load_raw_image, num_parallel_calls=AUTOTUNE).cache()
    preprocessed_ds = image_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    repeated_ds = preprocessed_ds.repeat(output_size * output_size).enumerate()
    occluded_ds = repeated_ds.map(
        lambda index, image: occlude_image(image,
                                           tf.convert_to_tensor(window_size, dtype=tf.int64),
                                           tf.convert_to_tensor(output_size, dtype=tf.int64),
                                           index)
    )
    occluded_ds = occluded_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return image_ds, preprocessed_ds, occluded_ds


# Loads a already trained model
def create_model():
    model = tf.keras.applications.resnet50.ResNet50(weights=str(RESNET50))
    model.summary()
    return model


def create_heat_map(model: tf.keras.Model,
                    preprocessed_ds: tf.data.Dataset,
                    occluded_ds: tf.data.Dataset,
                    output_size: int) -> tf.Tensor:
    # Evaluate the model with the original (not occluded) image
    image = next(iter(preprocessed_ds))
    output = model(tf.expand_dims(image, axis=0))
    # Original category/probability
    category = tf.argmax(tf.squeeze(output), axis=-1)
    probability = tf.gather(output, indices=category, axis=-1)
    # Computes sensitivity for all occluded regions
    sensitivity = []
    for batch in occluded_ds:
        sensitivity.append(tf.abs(tf.gather(model(batch), indices=category, axis=-1) - probability))
    return tf.reshape(tf.concat(sensitivity, axis=0), shape=(output_size, output_size))


def plot_heat_map(image: tf.Tensor, heatmap: tf.Tensor):
    # Setups figure
    fig = plt.figure()
    axes = fig.subplots(1, 3)
    # Plots original image
    axes[1].imshow(image)
    # Plots heatmap
    axes[0].imshow(heatmap.numpy())
    # Plots image with the transparency modified according to heatmap
    alpha = 255 * (tf.image.resize(tf.expand_dims(heatmap, axis=-1), [IMG_HEIGHT, IMG_WIDTH], antialias=True)
                   / tf.reduce_max(heatmap))
    axes[2].imshow(tf.concat((image, tf.cast(alpha, dtype=image.dtype)), axis=-1))
    # Shows magic!
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    plt.savefig(OUTPUTS / ('result_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'))
    plt.show()


def main(parser):
    # Parsing arguments
    args = parser.parse_args()
    # Allowing memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Creates model
    mdl = create_model()
    # Creates dataset one image repeated as many times as points will be in the heat map (output) with the correct
    # area occluded. The dataset provides batches to evaluate several images at once.
    image_ds, preprocessed_ds, occluded_ds = create_occlusion_dataset(args.image, args.window, args.output)
    # Creates heatmap (performing occlusion experiment)
    heatmap = create_heat_map(mdl, preprocessed_ds, occluded_ds, args.output)
    # Gets raw image
    image = tf.cast(next(iter(image_ds)), dtype=tf.int64)
    # Plot results
    plot_heat_map(image, heatmap)


if __name__ == '__main__':
    # Defines arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        help='Path to the image',
                        required=True)
    parser.add_argument('--window',
                        help='Size of the occlusion window (Default 20px -> Square 20x20)',
                        type=int,
                        default=20)
    parser.add_argument('--output',
                        help='Size of the output heatmap (Default 100px -> Square 100x100)',
                        type=int,
                        default=100)
    main(parser)
