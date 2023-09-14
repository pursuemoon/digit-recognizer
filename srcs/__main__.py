# -- coding: utf-8 --

from utils.log import logger

from ai.nn import Network
from ai.layer import LinearLayer, Conv2dLayer, PoolingLayer
from ai.calc import ActFunc, OptType, PoolType
from ai.data import MnistDataSet

# Just a fixed number is ok. Randomization is used in network initialization and data generation.
RANDOM_SEED = 1964

if __name__ == "__main__":

    logger.info("Building neural network...")

    network = Network(data_set=MnistDataSet(), random_seed=RANDOM_SEED)

    # network.add_layer(LinearLayer(input_dim=1*28*28, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(PoolingLayer(input_shape=(1,28,28), window_size=1, stride=1, pool_type=PoolType.Max))
    # network.add_layer(LinearLayer(input_dim=1*28*28, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=8, stride=3, padding=1, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=8*9*9, output_dim=10, act_func=ActFunc.Softmax))

    network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=7, filter_num=6, stride=2, padding=1, act_func=ActFunc.Relu))
    network.add_layer(Conv2dLayer(input_shape=(6,12,12), kernel_size=4, filter_num=8, stride=2, padding=0, act_func=ActFunc.Relu))
    network.add_layer(LinearLayer(input_dim=8*5*5, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=8, stride=2, padding=1, act_func=ActFunc.Relu))
    # network.add_layer(PoolingLayer(input_shape=(8,13,13), window_size=2, stride=2, pool_type=PoolType.Max))
    # network.add_layer(LinearLayer(input_dim=8*6*6, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=3, filter_num=10, stride=2, padding=1, act_func=ActFunc.Relu))
    # network.add_layer(PoolingLayer(input_shape=(10,14,14), window_size=2, stride=2, pool_type=PoolType.Max))
    # network.add_layer(Conv2dLayer(input_shape=(10,7,7), kernel_size=3, filter_num=20, stride=1, padding=0, act_func=ActFunc.Relu))
    # network.add_layer(PoolingLayer(input_shape=(20,5,5), window_size=2, stride=2, pool_type=PoolType.Max))
    # network.add_layer(LinearLayer(input_dim=20*2*2, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=6, stride=2, padding=1, act_func=ActFunc.Relu))
    # network.add_layer(PoolingLayer(input_shape=(6,13,13), window_size=2, stride=2, pool_type=PoolType.Average))
    # network.add_layer(LinearLayer(input_dim=6*6*6, output_dim=10, act_func=ActFunc.Softmax))

    # network.load_from_file('Pre-[[Conv-3k-10f-2s-1p-Relu]-[Pool]-2w-2s-[Conv-3k-20f-1s-0p-Relu]-[Pool]-2w-2s-[Linear-10-Softmax]]-[-1E-10s-Adam-0.001-0.0R]-[0.99m-0.999r-1e-08e].npz')

    kwargs = {
        'print_mean_square_error': False,
        'print_cross_entropy': False,
        'print_variance': False,
    }
    network.train(max_epoch=30, learning_rate=0.001, regular_coef=0.000,
                  opt_type=OptType.Adam, batch_size=10, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-8,
                  **kwargs)
    network.save_as_file(file_name=None, auto_name=True)

    network.test()
