# -- coding: utf-8 --

from utils.log import logger

from ai.nn import Network
from ai.layer import LinearLayer, Conv2dLayer, PoolingLayer, DropoutLayer
from ai.calc import ActFunc, OptType, PoolType
from ai.data import MnistDataSet, KaggleMnistDataSet

# Just a fixed number is ok. Randomization is used in network initialization and data generation.
RANDOM_SEED = 1964

if __name__ == "__main__":

    logger.info("Building neural network...")

    network = Network(data_set=MnistDataSet(), random_seed=RANDOM_SEED)

    # network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=10, stride=2, padding=1, act_func=ActFunc.Relu))
    # network.add_layer(Conv2dLayer(input_shape=(10,13,13), kernel_size=3, filter_num=20, stride=2, padding=0, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=20*6*6, output_dim=256, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=256, output_dim=64, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=64, output_dim=10, act_func=ActFunc.Softmax))

    network.load_from_file('mnist-cnn.npz')

    # kwargs = {
    #     'print_mean_square_error': False,
    #     'print_cross_entropy': True,
    #     'print_variance': False,
    # }
    # network.train(max_epoch=30, learning_rate=0.001, regular_coef=0.001,
    #               opt_type=OptType.Adam, batch_size=10, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-8,
    #               **kwargs)
    # network.save_as_file(file_name=None, auto_name=True)

    network.test(False)
