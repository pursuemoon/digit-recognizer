# -- coding: utf-8 --

from utils.log import logger

from ai.nn import Network, Layer, ActFunc, OptType
from ai.data import MnistDataSet

if __name__ == "__main__":

    logger.info("Building neural network...")

    network = Network(data_set=MnistDataSet(), random_seed=1964)
    network.add_layer(Layer(28 * 28, 32, ActFunc.Relu))
    network.add_layer(Layer(32, 10, ActFunc.Softmax))

    # network = Network(data_set=MnistDataSet())
    # network.load_from_file()

    kwargs = {
        'print_mean_square_error': False,
        'print_cross_entropy': False,
        'print_variance': False,
    }
    network.train(max_epoch=5, learning_rate=0.001, opt_type=OptType.Adam,
                  batch_size=10, momentum_coef=0.9, rms_coef=0.999, epsilon=1e-8,
                  **kwargs)
    # network.save_as_file()

    logger.info("Testing...")
    correct_rate = network.test()
    logger.info("correct_rate=%f" % correct_rate)
