# -- coding: utf-8 --

import os
import numpy
import gzip

from utils.env import Env
from utils.log import logger

from ai import nn

# Directory of MNIST examples.
DATA_DIR = "mnist"

# 60000 examples.
TRAIN_IMAGE_FILE = "train-images-idx3-ubyte.gz"
TRAIN_LABEL_FILE = "train-labels-idx1-ubyte.gz"

# 10000 examples.
TEST_IMAGE_FILE = "t10k-images-idx3-ubyte.gz"
TEST_LABEL_FILE = "t10k-labels-idx1-ubyte.gz"

def load_labels(label_path, size):
    labels = None
    with gzip.open(label_path, 'rb') as fh:
        contents = fh.read()
        labels = numpy.frombuffer(contents, numpy.uint8, size, 8)
        logger.debug('Labels loaded: shape={}'.format(labels.shape))
    return labels

def load_images(image_path, size):
    images = None
    with gzip.open(image_path, 'rb') as fh:
        contents = fh.read()
        images = numpy.frombuffer(contents, numpy.uint8, size * 28 * 28, 16)
        images = images.reshape(size, 28 * 28)
        logger.debug('Images loaded: shape={}'.format(images.shape))
    return images

def normalize(images):
    normal_images = numpy.array(images / 255, numpy.float64)
    return normal_images

if __name__ == "__main__":
    project_dir = Env.get_project_dir()
    logger.info("project_dir=" + project_dir)

    train_image_path = os.path.join(project_dir, DATA_DIR, TRAIN_IMAGE_FILE)
    train_label_path = os.path.join(project_dir, DATA_DIR, TRAIN_LABEL_FILE)
    test_image_path = os.path.join(project_dir, DATA_DIR, TEST_IMAGE_FILE)
    test_label_path = os.path.join(project_dir, DATA_DIR, TEST_LABEL_FILE)

    logger.info("Loading train data...")
    train_labels = load_labels(train_label_path, 60000)
    train_images = load_images(train_image_path, 60000)

    logger.info("Building neural network...")

    network = nn.Network(random_seed=1964)
    network.add_layer(nn.Layer(28 * 28, 32, nn.ActFunc.Sigmoid))
    network.add_layer(nn.Layer(32, 10, nn.ActFunc.Sigmoid))

    # network = nn.Network()
    # network.load_from_file('2023-08-17_00-34-56.npz')

    kwargs = {
        'print_mse': False,
        'print_var': False,
    }
    network.train(normalize(train_images), train_labels, 0.1, 5, **kwargs)
    # network.save_as_file()

    logger.info("Loading test data...")
    test_labels = load_labels(test_label_path, 10000)
    test_images = load_images(test_image_path, 10000)

    logger.info("Testing...")
    correct_rate = network.test(normalize(test_images), test_labels)
    logger.info("correct_rate=%f" % correct_rate)