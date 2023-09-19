# -- coding: utf-8 --

import os
import numpy
import gzip
import csv

import matplotlib.pyplot as plt

from utils.env import Env
from utils.log import logger


class AbstractDataSet(object):
    def __init__(self, name=None):
        self.name = name

    def _normalize(self, images):
        # Normalize input data.
        return images

    def _onehot_encode(self, labels, output_dim):
        # Encode labels in one-hot way.
        if len(labels) > 0:
            onehot_labels = numpy.zeros((len(labels), output_dim), numpy.float64)
            onehot_labels[numpy.arange(len(labels)), labels] = 1
            return onehot_labels
        else:
            return []

    def _load_train_data(self):
        return [], []

    def _load_test_data(self):
        return [], []

    def data_generator(self, batch_size, data_type='train', normalize=True, onehot=True, data_num=None, random_seed=None):
        if data_type == 'train':
            images, labels = self._load_train_data()
            assert len(images) == len(labels), "len(images) must be equal to len(labels) when generating train data"
        else:
            images, labels = self._load_test_data()
            if len(labels) == 0:
                logger.warn('This test data has no label.')

        if random_seed:
            numpy.random.seed(random_seed)
            logger.info('Seed was set to data_generator: random_seed=%d' % random_seed)

        index_list = numpy.array(range(len(images)))
        if data_num:
            index_list = index_list[:min(data_num, len(index_list))]
        if data_type == 'train':
            numpy.random.shuffle(index_list)

        img_list = []
        lbl_list = []
        for i in index_list:
            img_list.append(images[i])
            if len(labels) > i:
                lbl_list.append(labels[i]) 

            if batch_size > 0 and len(img_list) == batch_size:
                if normalize:
                    ret_imgs = self._normalize(numpy.array(img_list))
                else:
                    ret_imgs = numpy.array(img_list) 

                if onehot:
                    ret_lbls = self._onehot_encode(numpy.array(lbl_list), 10)
                else:
                    ret_lbls = numpy.array(lbl_list)

                yield ret_imgs, ret_lbls

                img_list = []
                lbl_list = []
        if len(img_list) > 0:
            if normalize:
                ret_imgs = self._normalize(numpy.array(img_list))
            else:
                ret_imgs = numpy.array(img_list) 

            if onehot:
                ret_lbls = self._onehot_encode(numpy.array(lbl_list), 10)
            else:
                ret_lbls = numpy.array(lbl_list)

            yield ret_imgs, ret_lbls

class MnistDataSet(AbstractDataSet):

    # Directory of MNIST examples.
    DATA_DIR = "data/mnist"

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
        super().__init__(name='mnist')
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

    def _normalize(self, images):
        normal_images = numpy.array(images / 255, numpy.float64)
        return normal_images

    def _load_train_data(self):
        train_labels = self.__load_labels(self.train_label_path, MnistDataSet.TRAIN_SIZE)
        train_images = self.__load_images(self.train_image_path, MnistDataSet.TRAIN_SIZE)
        return train_images, train_labels

    def _load_test_data(self):
        test_labels = self.__load_labels(self.test_label_path, MnistDataSet.TEST_SIZE)
        test_images = self.__load_images(self.test_image_path, MnistDataSet.TEST_SIZE)
        return test_images, test_labels

    def show_pictures(self, img_list, lbl_list, ret_list=None):
        dm = divmod(len(img_list), 10)
        rows = dm[0] + (0 if dm[1] == 0 else 1)
        for i in range(len(img_list)):
            plt.subplot(rows, 10, i+1)
            plt.imshow(X=img_list[i].reshape(28,28), cmap='gray')
            if len(lbl_list) > 0:
                if ret_list is not None and i < len(ret_list):
                    plt.title('{}-{}'.format(lbl_list[i], ret_list[i]))
                else:
                    plt.title('{}'.format(lbl_list[i]))
            plt.axis('off')
        plt.subplots_adjust(wspace=0.4, hspace=0.8)
        plt.show()


class KaggleMnistDataSet(AbstractDataSet):

    # Directory of Kaggle MNIST examples.
    DATA_DIR = "data/kaggle-mnist"

    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"

    DIMENSIONS = (1, 28, 28)
    TRAIN_SIZE = 42000
    TEST_SIZE = 28000

    def __init__(self):
        super().__init__(name='kaggle-mnist')
        project_dir = Env.get_project_dir()
        self.train_csv_path = os.path.join(project_dir, KaggleMnistDataSet.DATA_DIR, KaggleMnistDataSet.TRAIN_CSV)
        self.test_csv_path = os.path.join(project_dir, KaggleMnistDataSet.DATA_DIR, KaggleMnistDataSet.TEST_CSV)

    def __load_train_rows(self):
        imgs = []
        lbls = []
        with open(self.train_csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            idx = 0
            for row in csv_reader:
                if idx >= 1:
                    lbls.append(numpy.uint8(row[0]))
                    imgs.append(numpy.array(row[1:785], dtype=numpy.uint8))
                idx += 1
        return imgs, lbls

    def __load_test_rows(self):
        imgs = []
        with open(self.test_csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            idx = 0
            for row in csv_reader:
                if idx >= 1:
                    imgs.append(numpy.array(row, dtype=numpy.uint8))
                idx += 1
        return imgs

    def _normalize(self, images):
        normal_images = numpy.array(images / 255, numpy.float64)
        return normal_images

    def _load_train_data(self):
        return self.__load_train_rows()

    def _load_test_data(self):
        return self.__load_test_rows(), []

    def show_pictures(self, img_list, lbl_list, ret_list=None):
        dm = divmod(len(img_list), 10)
        rows = dm[0] + (0 if dm[1] == 0 else 1)
        for i in range(len(img_list)):
            plt.subplot(rows, 10, i+1)
            plt.imshow(X=img_list[i].reshape(28,28), cmap='gray')
            if len(lbl_list) > 0:
                if ret_list is not None and i < len(ret_list):
                    plt.title('{}-{}'.format(lbl_list[i], ret_list[i]))
                else:
                    plt.title('{}'.format(lbl_list[i]))
            plt.axis('off')
        plt.subplots_adjust(wspace=0.4, hspace=0.8)
        plt.show()


if __name__ == '__main__':
    data_sets = [MnistDataSet(), KaggleMnistDataSet()]
    for data_set in data_sets:
        data_generator = data_set.data_generator(batch_size=20, data_type='test',
                                                 normalize=False, onehot=False, data_num=100)
        for idx, data in enumerate(data_generator):
            x_train = data[0]
            y_train = data[1]
            data_set.show_pictures(x_train, y_train, y_train)
            break