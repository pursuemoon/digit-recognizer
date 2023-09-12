# -- coding: utf-8 --

from utils.log import logger

from ai.nn import Network
from ai.layer import LinearLayer, Conv2dLayer
from ai.calc import ActFunc, OptType
from ai.data import MnistDataSet

# Just a fixed number is ok. Randomization is used in network initialization and data generation.
RANDOM_SEED = 1964

if __name__ == "__main__":

    logger.info("Building neural network...")

    network = Network(data_set=MnistDataSet(), random_seed=RANDOM_SEED)
    network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=8, stride=2, padding=1, act_func=ActFunc.Relu))
    network.add_layer(Conv2dLayer(input_shape=(8,13,13), kernel_size=3, filter_num=5, stride=2, act_func=ActFunc.Relu))
    network.add_layer(LinearLayer(5 * 6 * 6, 10, ActFunc.Softmax))

    # network.add_layer(LinearLayer(28 * 28, 32, ActFunc.Relu))
    # network.add_layer(LinearLayer(32, 10, ActFunc.Softmax))

    # network = Network(data_set=MnistDataSet(), random_seed=RANDOM_SEED)
    # network.load_from_file()

    kwargs = {
        'print_mean_square_error': False,
        'print_cross_entropy': False,
        'print_variance': False,
    }
    network.train(max_epoch=3, learning_rate=0.1, regular_coef=0.000,
                  opt_type=OptType.MiniBatch, batch_size=100, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-8,
                  **kwargs)
    # network.save_as_file(auto_name=True)

    logger.info("Testing...")
    correct_rate = network.test()
    logger.info("correct_rate=%f" % correct_rate)
