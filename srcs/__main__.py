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
    # network = Network(data_set=KaggleMnistDataSet(), random_seed=RANDOM_SEED)

    # network.add_layer(LinearLayer(input_dim=1*28*28, output_dim=64, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=64, output_dim=64, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=64, output_dim=64, act_func=ActFunc.Relu))
    # network.add_layer(DropoutLayer(dropout_prob=0.2))
    # network.add_layer(LinearLayer(input_dim=64, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(PoolingLayer(input_shape=(1,28,28), window_size=7, stride=7, pool_type=PoolType.Max))
    # network.add_layer(LinearLayer(input_dim=1*4*4, output_dim=10, act_func=ActFunc.Softmax))

    network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=5, stride=3, padding=1, act_func=ActFunc.Relu))
    network.add_layer(LinearLayer(input_dim=5*9*9, output_dim=128, act_func=ActFunc.Relu))
    network.add_layer(DropoutLayer(dropout_prob=0.4))
    network.add_layer(LinearLayer(input_dim=128, output_dim=128, act_func=ActFunc.Relu))
    network.add_layer(DropoutLayer(dropout_prob=0.1))
    network.add_layer(LinearLayer(input_dim=128, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=10, stride=2, padding=1, act_func=ActFunc.Relu))
    # network.add_layer(Conv2dLayer(input_shape=(10,13,13), kernel_size=3, filter_num=20, stride=2, padding=0, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=20*6*6, output_dim=256, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=256, output_dim=64, act_func=ActFunc.Relu))
    # network.add_layer(LinearLayer(input_dim=64, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=3, filter_num=10, stride=2, padding=1, act_func=ActFunc.Relu))
    # network.add_layer(PoolingLayer(input_shape=(10,14,14), window_size=2, stride=2, pool_type=PoolType.Max))
    # network.add_layer(Conv2dLayer(input_shape=(10,7,7), kernel_size=3, filter_num=20, stride=1, padding=0, act_func=ActFunc.Relu))
    # network.add_layer(PoolingLayer(input_shape=(20,5,5), window_size=2, stride=2, pool_type=PoolType.Max))
    # network.add_layer(LinearLayer(input_dim=20*2*2, output_dim=10, act_func=ActFunc.Softmax))

    # network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=20, stride=2, padding=1, act_func=ActFunc.Relu))
    # network.add_layer(PoolingLayer(input_shape=(20,13,13), window_size=2, stride=2, pool_type=PoolType.Max))
    # network.add_layer(LinearLayer(input_dim=20*6*6, output_dim=10, act_func=ActFunc.Softmax))

    # 97.90
    # network.load_from_file('[[Conv-5k-10f-3s-1p-Relu]-[Linear-128-Relu]-[Linear-128-Relu]-[Linear-10-Softmax]]-[200E-10s-Adam-0.001-0.001R]-[0.99m-0.999r-1e-08e].npz')

    # 98.97
    # network.load_from_file('[[Conv-5k-5f-3s-1p-Relu]-[Linear-128-Relu]-[Linear-128-Relu]-[Linear-10-Softmax]]-[100E-10s-RmsProp-0.001-0.001R]-[0.999r-1e-08e].npz')

    # 99.13
    # network.load_from_file('[[Conv-5k-5f-3s-1p-Relu]-[Linear-128-Relu]-[Linear-128-Relu]-[Linear-10-Softmax]]-[100E-10s-Adam-0.001-0.001R]-[0.99m-0.999r-1e-08e].npz')

    # 98.74
    # network.load_from_file('[[Conv-5k-10f-2s-1p-Relu]-[Conv-3k-20f-2s-0p-Relu]-[Linear-256-Relu]-[Linear-64-Relu]-[Linear-10-Softmax]]-[50E-10s-Adam-0.001-0.001R]-[0.99m-0.999r-1e-08e].npz')

    # 99.05
    # network.load_from_file('Pre-[[Conv-5k-10f-3s-1p-Relu]-[Linear-128-Relu]-[Linear-128-Relu]-[Linear-10-Softmax]]-[3E-10s-Adam-0.001-0.001R]-[0.99m-0.999r-1e-08e].npz')

    # 99.22
    # network.load_from_file('Pre-[[Conv-5k-10f-2s-1p-Relu]-[Conv-3k-20f-2s-0p-Relu]-[Linear-256-Relu]-[Linear-64-Relu]-[Linear-10-Softmax]]-[2E-10s-Adam-0.001-0.001R]-[0.99m-0.999r-1e-08e].npz')

    kwargs = {
        'print_mean_square_error': False,
        'print_cross_entropy': True,
        'print_variance': False,
    }
    network.train(max_epoch=200, learning_rate=0.001, regular_coef=0.001,
                  opt_type=OptType.Adam, batch_size=10, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-8,
                  **kwargs)
    # network.save_as_file(file_name=None, auto_name=True)

    network.test(False)
