#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import datetime
import os

# Constants
DATASETS = pathlib.Path(os.environ['DATASETS'])
OUTPUTS = pathlib.Path(os.environ['OUTPUTS'])
RESNET50 = pathlib.Path(os.environ['WEIGHTS']) / 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
RESNET50_NOTOP = pathlib.Path(os.environ['WEIGHTS']) / 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
BUFFER_SIZE = BATCH_SIZE * 32


# Loads raw image
def load_raw_image(path: tf.Tensor) -> tf.Tensor:
    serialized = tf.io.read_file(path)
    image = tf.io.decode_image(serialized, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], antialias=True)
    return image


# Preprocess image according to pretrained network
def preprocess_image(img: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.resnet50.preprocess_input(img)


# Loads a already trained model
def create_model():
    # Creates a dense layer with same weights as pretrained resnet but without softmax activation
    resnet = tf.keras.applications.resnet50.ResNet50(weights=str(RESNET50))
    average = resnet.layers[-2]
    dense = tf.keras.layers.Dense(1000, activation=None, trainable=False)
    dense.build(average.output_shape)
    dense.set_weights(resnet.layers[-1].get_weights())

    # Loads a resnet pretrained model without dense layers
    resnet = tf.keras.applications.resnet50.ResNet50(weights=str(RESNET50_NOTOP), include_top=False)
    for index in range(len(resnet.layers)):
        resnet.layers[index].trainable = False

    # Builds final model combining the resnet without top and the dense layer created before
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    features = resnet(inputs)
    scores = dense(average(features))
    model = tf.keras.Model(inputs=inputs, outputs=scores)
    model.summary()

    return model


def create_saliency_map(image: tf.Tensor, model: tf.keras.Model) -> tf.Tensor:
    image = preprocess_image(image)
    with tf.GradientTape() as tape:
        tape.watch(image)
        output = model(tf.expand_dims(image, axis=0))
        index = tf.argmax(output, axis=-1)
        prediction = tf.gather(output, index, axis=-1)
    gradients = tape.gradient(prediction, [image])
    return tf.squeeze(tf.reduce_max(tf.abs(gradients), axis=-1))


def plot_saliency_map(image: tf.Tensor, saliency: tf.Tensor):
    # Setups figure
    fig = plt.figure()
    axes = fig.subplots(1, 3)
    # Plots original image
    axes[1].imshow(image)
    # Plots saliency map
    axes[0].imshow(saliency.numpy())
    # Plots image with the transparency modified according to saliency
    alpha = 255 * tf.expand_dims(saliency, axis=-1) / tf.reduce_max(saliency)
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
    # Loads and preprocess image
    image = load_raw_image(args.image)
    # Builds saliency map
    saliency = create_saliency_map(image, mdl)
    # Plots saliency map
    plot_saliency_map(tf.cast(image, dtype=tf.int64), saliency)


if __name__ == '__main__':
    # Defines arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        help='Path to the image',
                        required=True)
    main(parser)
