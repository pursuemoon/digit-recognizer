# -- coding: utf-8 --

import os
import numpy
import tqdm
import time
import signal

from utils.log import logger
from utils.env import Env
from utils.time import human_readable_time
from utils.file import prepare_directory

from ai.layer import LayerType, LinearLayer, Conv2dLayer, PoolingLayer
from ai.calc import ActFunc, PoolType, OptType, Optimizer


# Neural Network.

class Network(object):
    def __init__(self, data_set, random_seed=None):
        self.data_set = data_set
        self.layers = []
        self.optimizers = []
        self.is_trained = False
        self.is_being_stoped = False
        if random_seed:
            numpy.random.seed(random_seed)
            logger.info('Seed was set to Network: random_seed=%d' % random_seed)
        signal.signal(signal.SIGINT, self.__sigint_handler)

    def add_layer(self, layer):
        logger.info('Adding layer-{}: {}'.format(len(self.layers) + 1, layer.get_abstract()))
        self.layers.append(layer)

    def __validate(self, nn_input_dimensions):
        layer_size = len(self.layers)
        assert layer_size > 0, "At least 1 layer is needed"

        if self.layers[0].type == LayerType.Linear:
            assert numpy.prod(nn_input_dimensions) == self.layers[0].input_dim, "Invalid input dim of first layer"
        elif self.layers[0].type == LayerType.Conv2d:
            assert nn_input_dimensions == self.layers[0].input_shape, "Invalid input shape of first layer"
        elif self.layers[0].type == LayerType.Pooling:
            assert nn_input_dimensions == self.layers[0].input_shape, "Invalid input shape of first layer"
        else:
            logger.error("Invalid layer type of first layer: {}".format(self.layers[0].type.name))
            assert False

        for i in range(layer_size):
            layer = self.layers[i]
            if i > 0:
                last_layer = self.layers[i - 1]
                if last_layer.type == LayerType.Linear and layer.type == LayerType.Linear:
                    assert last_layer.output_dim == layer.input_dim, \
                        "input_dim of No.{} layer [{}] must be equal to output_dim of last layer [{}]".format(i+1, layer.type.name, last_layer.type.name)
                elif last_layer.type == LayerType.Conv2d and layer.type == LayerType.Conv2d:
                    assert last_layer.filter_num == layer.input_shape[0], \
                        "input_shape[0] of No.{} layer [{}] must be equal to filter_num of last layer [{}]".format(i+1, layer.type.name, last_layer.type.name)
                elif last_layer.type == LayerType.Conv2d and layer.type == LayerType.Pooling:
                    assert last_layer.output_shape == layer.input_shape, \
                        "input_shape of No.{} layer [{}] must be equal to output_shape of last layer [{}]".format(i+1, layer.type.name, last_layer.type.name)
                elif last_layer.type == LayerType.Conv2d and layer.type == LayerType.Linear:
                    assert last_layer.output_dim == layer.input_dim, \
                        "input_dim of No.{} layer [{}] must be equal to output_dim of last layer [{}]".format(i+1, layer.type.name, last_layer.type.name)
                elif last_layer.type == LayerType.Pooling and layer.type == LayerType.Linear:
                    assert last_layer.output_dim == layer.input_dim, \
                        "input_dim of No.{} layer [{}] must be equal to output_dim of last layer [{}]".format(i+1, layer.type.name, last_layer.type.name)
                elif last_layer.type == LayerType.Pooling and layer.type == LayerType.Conv2d:
                    assert last_layer.output_shape == layer.input_shape, \
                        "input_shape of No.{} layer [{}] must be equal to output_shape of last layer [{}]".format(i+1, layer.type.name, last_layer.type.name)
                else:
                    assert False, 'Invalid combination: [{}] -> [{}]'.format(last_layer.type.name, layer.type.name)
            if i < layer_size - 1:
                assert layer.act_func != ActFunc.Softmax, "Softmax is only allowed as activation function on the last layer"

    def __sigint_handler(self, signum, frame):
        self.is_being_stoped = True
        self.optimizers[-1].mark_not_finish()
        logger.info('Training is being early stoped.')

    def train(self, max_epoch, learning_rate, regular_coef=0, opt_type=OptType.MiniBatch,
              batch_size=20, momentum_coef=0.9, rms_coef=0.999, epsilon=1e-7,
              **kwargs):
        self.__validate(self.data_set.DIMENSIONS)

        # Initialize the current optimizer and its associated values.
        optimizer = Optimizer(opt_type=opt_type, max_epoch=max_epoch, learning_rate=learning_rate, batch_size=batch_size,
                              regular_coef=regular_coef, momentum_coef=momentum_coef, rms_coef=rms_coef, epsilon=epsilon)
        self.optimizers.append(optimizer)
        for layer in self.layers:
            layer.init_optimization_value(optimizer)

        self.is_trained = False

        logger.info('Training started: {}'.format(optimizer))
        start_time = time.time()

        layer_size = len(self.layers)

        print_mean_square_error = kwargs['print_mean_square_error'] if 'print_mean_square_error' in kwargs else False
        print_cross_entropy = kwargs['print_cross_entropy'] if 'print_cross_entropy' in kwargs else False
        print_variance = kwargs['print_variance'] if 'print_variance' in kwargs else False

        total_cnt = max_epoch * self.data_set.TRAIN_SIZE
        process_bar = tqdm.tqdm(total=total_cnt, colour='cyan', ncols=120, unit_scale=True, desc="Training")

        step = 0
        for i in range(max_epoch):
            if self.is_being_stoped:
                break

            # Generate examples by batch randomly in each epoch.
            data_generator = self.data_set.data_generator(batch_size)

            for j, data in enumerate(data_generator):
                step += 1
                x_train = data[0]
                y_train = data[1]
                for k in range(layer_size):
                    if k == 0:
                        self.layers[k].calculate_forward(x_train, True)
                    else:
                        self.layers[k].calculate_forward(self.layers[k - 1].output, True)

                # Calculate the delta of each layer through recursion.
                for k in reversed(range(layer_size)):
                    last_layer = self.layers[k - 1] if k > 0 else None
                    final_answer = y_train if k == layer_size - 1 else None

                    self.layers[k].propagate_backward(last_layer, final_answer)
                
                # Update parameters of each layer.
                for k in reversed(range(layer_size)):
                    self.layers[k].update_parameters(learning_rate, optimizer, step)

                current_cnt = i * self.data_set.TRAIN_SIZE + j * batch_size + len(data)
                process_bar.update(len(data[0]))

                if j % 500 == 0 or len(data[0]) != batch_size:
                    with numpy.printoptions(linewidth=numpy.inf):
                        used_time = time.time() - start_time

                        index = j * batch_size + len(data[0])
                        need_time = human_readable_time(used_time / current_cnt * total_cnt - used_time)

                        extra_log = ''

                        if print_mean_square_error:
                            mse = numpy.mean(numpy.square(self.layers[layer_size - 1].output - y_train))
                            extra_log += ", mean_square_error={}".format(mse)

                        if print_cross_entropy:
                            cre = -numpy.mean(numpy.sum(y_train * numpy.log(self.layers[layer_size - 1].output), axis=1))
                            extra_log += ", cross_entropy={}".format(cre)

                        if print_variance:
                            output_var = [layer.get_output_var() for layer in self.layers]
                            extra_log += ", output_variance={}".format(output_var)

                        logger.debug("epoch={}, index={}, need_time={}{}".format(i + 1, index, need_time, extra_log))

                if self.is_being_stoped:
                    break

        process_bar.close()
        self.is_trained = True
        
        end_time = time.time()
        logger.info('Training ended. Time used: {}'.format(human_readable_time(end_time - start_time)))

    def test(self, case_num=None):
        assert self.is_trained, "neural network must be trained first before using it"

        start_time = time.time()
        case_num = self.data_set.TEST_SIZE if case_num is None else case_num
        logger.info("Test started: case_num={}".format(case_num))

        cnt_correct = 0
        layer_size = len(self.layers)

        data = self.data_set.load_test_data()

        x_test = self.data_set.normalize(data[0])[:case_num]
        y_test = self.data_set.onehot_encode(data[1])[:case_num]

        for k in range(layer_size):
            if k == 0:
                self.layers[k].calculate_forward(x_test, False)
            else:
                self.layers[k].calculate_forward(self.layers[k - 1].output, False)

        ans = numpy.argmax(self.layers[layer_size - 1].output, axis=1)
        std = numpy.argmax(y_test, axis=1)
        cnt_correct = numpy.sum(ans == std)
        correct_rate = cnt_correct / len(y_test)

        end_time = time.time()
        logger.info('Test ended. Time used: {}'.format(human_readable_time(end_time - start_time)))
        logger.info("correct_rate=%f" % correct_rate)

        return correct_rate

    def save_as_file(self, file_name=None, auto_name=False):
        if not file_name:
            if auto_name:
                pretrain_mark = 'Pre-' if len(self.optimizers) > 1 else ''
                layers = []
                for layer in self.layers:
                    if layer.type == LayerType.Linear:
                        layers.append('[Linear-{}-{}]'.format(layer.output_dim, layer.act_func.name))
                    if layer.type == LayerType.Conv2d:
                        layers.append('[Conv-{}k-{}f-{}s-{}p-{}]'.format(layer.kernel_size, layer.filter_num, layer.stride, layer.padding, layer.act_func.name))
                    if layer.type == LayerType.Pooling:
                        layers.append('[Pool-{}w-{}s]'.format(layer.window_size, layer.stride))
                structure = '-'.join(layers)
                optimizer = self.optimizers[-1]
                file_name = '{}[{}]-{}.npz'.format(pretrain_mark, structure, optimizer.as_short_name())
            else:
                file_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + ".npz"

        file_path = os.path.join(prepare_directory(Env.MODEL_DIR), file_name)

        model = {}

        # Save parameters of each layer.
        model['layer_size'] = len(self.layers)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            model['layer_%d_type' % (i + 1)] = layer.type
            if layer.type == LayerType.Linear:
                model['layer_%d_input_dim' % (i + 1)] = layer.input_dim
                model['layer_%d_output_dim' % (i + 1)] = layer.output_dim
                model['layer_%d_weights' % (i + 1)] = layer.weights
                model['layer_%d_bias' % (i + 1)] = layer.bias
                model['layer_%d_act_func' % (i + 1)] = layer.act_func.value
            if layer.type == LayerType.Conv2d:
                model['layer_%d_input_shape' % (i + 1)] = layer.input_shape
                model['layer_%d_kernel_size' % (i + 1)] = layer.kernel_size
                model['layer_%d_filter_num' % (i + 1)] = layer.filter_num
                model['layer_%d_stride' % (i + 1)] = layer.stride
                model['layer_%d_padding' % (i + 1)] = layer.padding
                model['layer_%d_weights' % (i + 1)] = layer.weights
                model['layer_%d_bias' % (i + 1)] = layer.bias
                model['layer_%d_act_func' % (i + 1)] = layer.act_func.value
            if layer.type == LayerType.Pooling:
                model['layer_%d_input_shape' % (i + 1)] = layer.input_shape
                model['layer_%d_window_size' % (i + 1)] = layer.window_size
                model['layer_%d_stride' % (i + 1)] = layer.stride
                model['layer_%d_pool_type' % (i + 1)] = layer.pool_type

        # Save training history.
        model['training_round'] = len(self.optimizers)
        for i in range(len(self.optimizers)):
            model['optimizer_at_training_round_%d' % (i + 1)] = self.optimizers[i].as_nd_array()

        numpy.savez(file_path, **model)
        logger.info('Parameters of neural network are saved as file: %s' % file_path)

    def load_from_file(self, file_name='default.npz'):
        file_path = os.path.join(prepare_directory(Env.MODEL_DIR), file_name)
        assert os.path.exists(file_path), "There is no such model file: file_path=%s" % file_path
        assert len(self.layers) == 0, "Model is loaded"

        model = numpy.load(file_path)

        layer_size = model['layer_size']
        for i in range(layer_size):
            type = model['layer_%d_type' % (i + 1)]

            if type == LayerType.Linear:
                input_dim = model['layer_%d_input_dim' % (i + 1)]
                output_dim = model['layer_%d_output_dim' % (i + 1)]

                weights = model['layer_%d_weights' % (i + 1)]
                bias = model['layer_%d_bias' % (i + 1)]
                act_func = ActFunc(model['layer_%d_act_func' % (i + 1)])

                self.add_layer(LinearLayer(input_dim=input_dim, output_dim=output_dim, act_func=act_func,
                                           weights=weights, bias=bias))
            if type == LayerType.Conv2d:
                input_shape = tuple(model['layer_%d_input_shape' % (i + 1)])
                kernel_size = model['layer_%d_kernel_size' % (i + 1)]
                filter_num = model['layer_%d_filter_num' % (i + 1)]
                stride = model['layer_%d_stride' % (i + 1)]
                padding = model['layer_%d_padding' % (i + 1)]

                weights = model['layer_%d_weights' % (i + 1)]
                bias = model['layer_%d_bias' % (i + 1)]
                act_func = ActFunc(model['layer_%d_act_func' % (i + 1)])

                self.add_layer(Conv2dLayer(input_shape=input_shape, kernel_size=kernel_size,
                                           filter_num=filter_num, stride=stride, padding=padding,
                                           act_func=act_func, weights=weights, bias=bias))

            if type == LayerType.Pooling:
                input_shape = tuple(model['layer_%d_input_shape' % (i + 1)])
                window_size = model['layer_%d_window_size' % (i + 1)]
                stride = model['layer_%d_stride' % (i + 1)]
                pool_type = PoolType(model['layer_%d_pool_type' % (i + 1)])

                self.add_layer(PoolingLayer(input_shape=input_shape, window_size=window_size, stride=stride, pool_type=pool_type))

        training_round = model['training_round']
        for i in range(training_round):
            opt_params = model['optimizer_at_training_round_%d' % (i + 1)]
            self.optimizers.append(Optimizer(nd_array=opt_params))
            logger.info('Pre-trained: {}'.format(self.optimizers[-1]))

        self.is_trained = True
        logger.info('Parameters were loaded.')
