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
from ai.calc import ActFunc, OptType, Optimizer


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
        else:
            assert False, "Invalid layer type of first layer"

        for i in range(layer_size):
            layer = self.layers[i]
            if i > 0:
                last_layer = self.layers[i - 1]
                if last_layer.type == LayerType.Linear and layer.type == LayerType.Linear:
                    assert last_layer.output_dim == layer.input_dim, \
                        "input_dim of current layer [{}] must be equal to output_dim of last layer [{}]".format(layer.type.name, last_layer.type.name)
                elif last_layer.type == LayerType.Conv2d and layer.type == LayerType.Conv2d:
                    assert last_layer.filter_num == layer.input_shape[0], \
                        "input_shape[0] of current layer [{}] must be equal to filter_num of last layer [{}]".format(layer.type.name, last_layer.type.name)
                elif last_layer.type == LayerType.Conv2d and layer.type == LayerType.Pooling:
                    logger.warn('Pass validation: [{}] -> [{}]'.format(last_layer.type.name, layer.type.name))
                elif last_layer.type == LayerType.Conv2d and layer.type == LayerType.Linear:
                    assert last_layer.output_dim == layer.input_dim, \
                        "input_dim of current layer [{}] must be equal to output_dim of last layer [{}]".format(layer.type.name, last_layer.type.name)
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
                        self.layers[k].calculate_forward(x_train)
                    else:
                        self.layers[k].calculate_forward(self.layers[k - 1].output)

                # Calculate the delta of each layer through recursion.
                for k in reversed(range(1, layer_size)):
                    last_layer = self.layers[k - 1]
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

    def test(self):
        assert self.is_trained, "neural network must be trained first before using it"

        cnt_correct = 0
        layer_size = len(self.layers)

        data = self.data_set.load_test_data()

        x_test = self.data_set.normalize(data[0])
        y_test = self.data_set.onehot_encode(data[1])

        for k in range(layer_size):
            if k == 0:
                self.layers[k].calculate_forward(x_test)
            else:
                self.layers[k].calculate_forward(self.layers[k - 1].output)

        ans = numpy.argmax(self.layers[layer_size - 1].output, axis=1)
        std = numpy.argmax(y_test, axis=1)
        cnt_correct = numpy.sum(ans == std)
        correct_rate = cnt_correct / len(y_test)

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
                        layers.append('[Conv-{}-{}-{}]'.format(layer.kernel_size, layer.filter_num, layer.act_func.name))
                    if layer.type == LayerType.Pooling:
                        layers.append('[Pool]')
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
                model['layer_%d_weights' % (i + 1)] = layer.weights
                model['layer_%d_bias' % (i + 1)] = layer.bias
                model['layer_%d_act_func' % (i + 1)] = layer.act_func.value
            if layer.type == LayerType.Conv2d:
                logger.warn('Conv2dLayer is not saved yet.')
            if layer.type == LayerType.Pooling:
                logger.warn('PoolingLayer is not saved yet.')

        # Save training history.
        model['training_round'] = len(self.optimizers)
        for i in range(len(self.optimizers)):
            model['optimizer_at_training_round_%d' % (i + 1)] = self.optimizers[i].as_nd_array()

        numpy.savez(file_path, **model)
        logger.info('Parameters of neural network are saved as file: %s' % file_path)

    def load_from_file(self, file_name='default.npz'):
        file_path = os.path.join(prepare_directory(Env.MODEL_DIR), file_name)
        assert os.path.exists(file_path), "there is no such model file: file_path=%s" % file_path

        model = numpy.load(file_path)

        layer_size = model['layer_size']
        for i in range(layer_size):
            weights = model['layer_%d_weights' % (i + 1)]
            bias = model['layer_%d_bias' % (i + 1)]
            act_func = ActFunc(model['layer_%d_act_func' % (i + 1)])
            self.add_layer(LinearLayer(weights.shape[0], weights.shape[1], act_func, weights, bias))

        training_round = model['training_round']
        for i in range(training_round):
            opt_params = model['optimizer_at_training_round_%d' % (i + 1)]
            self.optimizers.append(Optimizer(nd_array=opt_params))
            logger.info('Pre-training was done: {}'.format(self.optimizers[-1]))

        self.is_trained = True
        logger.info('Parameters were loaded.')
