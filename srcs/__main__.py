# -- coding: utf-8 --

from utils.log import logger

from ai.nn import Network, Layer, ActFunc, OptType
from ai.data import MnistDataSet

# Just a fixed number is ok. Randomization is used in network initialization and data generation.
RANDOM_SEED = 1964

if __name__ == "__main__":

    logger.info("Building neural network...")

    network = Network(data_set=MnistDataSet(), random_seed=RANDOM_SEED)
    network.add_layer(Layer(28 * 28, 64, ActFunc.Relu))
    network.add_layer(Layer(64, 32, ActFunc.Relu))
    network.add_layer(Layer(32, 10, ActFunc.Softmax))

    # network = Network(data_set=MnistDataSet(), random_seed=RANDOM_SEED)
    # network.load_from_file()

    kwargs = {
        'print_mean_square_error': False,
        'print_cross_entropy': False,
        'print_variance': False,
    }
    network.train(max_epoch=20, learning_rate=0.1, opt_type=OptType.Momentum,
                  batch_size=10, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-8,
                  **kwargs)
    network.save_as_file(auto_name=True)

    logger.info("Testing...")
    correct_rate = network.test()
    logger.info("correct_rate=%f" % correct_rate)
