# -- coding: utf-8 --

import os
import numpy
import gzip

import matplotlib.pyplot as plt

from utils.env import Env
from utils.log import logger


class MnistDataSet(object):

    # Directory of MNIST examples.
    DATA_DIR = "mnist"

    # 60000 examples.
    TRAIN_IMAGE_FILE = "train-images-idx3-ubyte.gz"
    TRAIN_LABEL_FILE = "train-labels-idx1-ubyte.gz"

    # 10000 examples.
    TEST_IMAGE_FILE = "t10k-images-idx3-ubyte.gz"
    TEST_LABEL_FILE = "t10k-labels-idx1-ubyte.gz"

    DIMENSIONS = (1, 28, 28)
    TRAIN_SIZE = 60000
    TEST_SIZE = 10000

    def __init__(self):
        project_dir = Env.get_project_dir()
        self.train_image_path = os.path.join(project_dir, MnistDataSet.DATA_DIR, MnistDataSet.TRAIN_IMAGE_FILE)
        self.train_label_path = os.path.join(project_dir, MnistDataSet.DATA_DIR, MnistDataSet.TRAIN_LABEL_FILE)
        self.test_image_path = os.path.join(project_dir, MnistDataSet.DATA_DIR, MnistDataSet.TEST_IMAGE_FILE)
        self.test_label_path = os.path.join(project_dir, MnistDataSet.DATA_DIR, MnistDataSet.TEST_LABEL_FILE)

    def __load_labels(self, label_path, size):
        labels = None
        with gzip.open(label_path, 'rb') as fh:
            contents = fh.read()
            labels = numpy.frombuffer(contents, numpy.uint8, size, 8)
            logger.debug('Labels loaded: shape={}'.format(labels.shape))
        return labels

    def __load_images(self, image_path, size):
        images = None
        with gzip.open(image_path, 'rb') as fh:
            contents = fh.read()
            images = numpy.frombuffer(contents, numpy.uint8, size * 28 * 28, 16)
            images = images.reshape(size, 1 * 28 * 28)
            logger.debug('Images loaded: shape={}'.format(images.shape))
        return images

    def normalize(self, images):
        normal_images = numpy.array(images / 255, numpy.float64)
        return normal_images

    def onehot_encode(self, labels, output_dim=10):
        onehot_labels = numpy.zeros((len(labels), output_dim), numpy.float64)
        onehot_labels[numpy.arange(len(labels)), labels] = 1
        return onehot_labels

    def load_train_data(self):
        train_labels = self.__load_labels(self.train_label_path, 60000)
        train_images = self.__load_images(self.train_image_path, 60000)
        return train_images, train_labels

    def load_test_data(self):
        test_labels = self.__load_labels(self.test_label_path, 10000)
        test_images = self.__load_images(self.test_image_path, 10000)
        return test_images, test_labels

    def data_generator(self, batch_size, normalize=True, random_seed=None):
        images, labels = self.load_train_data()
        assert len(images) == len(labels), "len(images) must be equal to len(labels) when using data_generator"

        if random_seed:
            numpy.random.seed(random_seed)
            logger.info('Seed was set to data_generator: random_seed=%d' % random_seed)

        index_list = numpy.array(range(len(images)))
        numpy.random.shuffle(index_list)

        img_list = []
        lbl_list = []
        for i in index_list:
            img_list.append(images[i])
            lbl_list.append(labels[i])
            if batch_size > 0 and len(img_list) == batch_size:
                if normalize:
                    yield self.normalize(numpy.array(img_list)), self.onehot_encode(numpy.array(lbl_list), 10)
                else:
                    yield numpy.array(img_list), self.onehot_encode(numpy.array(lbl_list), 10)
                img_list = []
                lbl_list = []
        if len(img_list) > 0:
            if normalize:
                yield self.normalize(numpy.array(img_list)), self.onehot_encode(numpy.array(lbl_list), 10)
            else:
                yield numpy.array(img_list), self.onehot_encode(numpy.array(lbl_list), 10)

    def show_pictures(self, img_list, lbl_list, ret_list=None):
        assert len(img_list) == len(lbl_list)
        dm = divmod(len(img_list), 10)
        rows = dm[0] + (0 if dm[1] == 0 else 1)
        for i in range(len(img_list)):
            plt.subplot(rows, 10, i+1)
            plt.imshow(X=img_list[i].reshape(28,28), cmap='gray')
            if ret_list is not None and i < len(ret_list):
                plt.title('{}-{}'.format(lbl_list[i], ret_list[i]))
            else:
                plt.title('{}'.format(lbl_list[i]))
            plt.axis('off')
        plt.subplots_adjust(wspace=0.4, hspace=0.8)
        plt.show()


if __name__ == '__main__':
    data_set = MnistDataSet()
    data_generator = data_set.data_generator(50, False)
    for idx, data in enumerate(data_generator):
        x_train = data[0]
        y_train = data[1]
        data_set.show_pictures(x_train, numpy.argmax(y_train, axis=1), numpy.argmax(y_train, axis=1))
        break