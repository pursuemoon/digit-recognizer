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

from ai.calc import apply_activation, apply_activation_derivative, ActFunc


# Layer of Neural Network.

class Layer(object):
    def __init__(self, input_dim, output_dim, act_func=None, weights=None, bias=None, random_seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if random_seed:
            numpy.random.seed(random_seed)

        if act_func is not None:
            assert isinstance(act_func, ActFunc), "act_func must be of enumeration type of ActFunc"
            self.act_func = act_func
        else:
            self.act_func = ActFunc.Identity

        d_0 = apply_activation_derivative(apply_activation(0, self.act_func), self.act_func)
        xavier_gain = numpy.sqrt(2) if d_0 == 0 else 1 / d_0

        if weights is not None:
            assert input_dim == weights.shape[0], "weights.shape[0] must be equal to input_dim"
            assert output_dim == weights.shape[1], "weights.shape[1] must be equal to output_dim"
            self.weights = weights
        else:
            # 1. Xavier Uniform Initialization
            # limit = numpy.sqrt(6 / (input_dim + output_dim))
            # self.weights = numpy.random.uniform(-limit, limit, size=(input_dim, output_dim)) * xavier_gain

            # 2. Xavier Normal Initialization
            self.weights = numpy.random.normal(0, numpy.sqrt(2 / (input_dim + output_dim)), size=(input_dim, output_dim)) * xavier_gain

        if bias is not None:
            assert output_dim == bias.shape[0], "bias.shape[0] must be equal to output_dim"
            self.bias = bias
        else:
            # 1. Xavier Uniform Initialization
            # limit = numpy.sqrt(6 / (input_dim + output_dim))
            # self.bias = numpy.random.uniform(-limit, limit, size=output_dim) * xavier_gain

            # 2. Xavier Normal Initialization
            self.bias = numpy.random.normal(0, numpy.sqrt(2 / (input_dim + output_dim)), size=output_dim) * xavier_gain

        self.output = None
        self.error = None
        self.delta = None

    def calculate_forward(self, x):
        # Calcuate outputs on each layer.
        tmp = numpy.dot(x, self.weights) + self.bias
        self.output = apply_activation(tmp, self.act_func)

    def propagate_backward(self, next_layer, y_output):
        # Calculate errors from the next layer.
        if not next_layer:
            self.error = self.output - y_output
        else:
            self.error = numpy.dot(next_layer.delta, next_layer.weights.T)

        # Calculate deltas of this layer.
        self.delta = self.error * apply_activation_derivative(self.output, self.act_func)
    
    def update_parameters(self, learning_rate, batch_size, last_layer, x_input):
        # Update parameters by back-propogation.
        layer_input = last_layer.output if last_layer else x_input

        gradient_sum = numpy.zeros_like(self.weights)
        for idx in range(len(layer_input)):
            gradient_sum += numpy.atleast_2d(layer_input[idx]).T * self.delta[idx]
        self.weights -= learning_rate * gradient_sum / batch_size

        self.bias -= learning_rate * numpy.mean(self.delta, axis=0)

    def get_output_std(self):
        return numpy.std(self.output)

    def get_output_var(self):
        return numpy.var(self.output)


# Neural Network.

class Network(object):
    def __init__(self, data_set, random_seed=None):
        self.data_set = data_set
        self.layers = []
        self.is_trained = False
        self.is_being_stoped = False
        if random_seed:
            numpy.random.seed(random_seed)
            logger.info('Seed was set to Network: random_seed=%d' % random_seed)
        signal.signal(signal.SIGINT, self.__sigint_handler)

    def add_layer(self, layer):
        logger.info('Adding layer-{}: weights.shape={}, act_func={}'.format(len(self.layers) + 1, layer.weights.shape, layer.act_func.name))
        self.layers.append(layer)

    def __validate(self, nn_input_dim):
        layer_size = len(self.layers)
        assert layer_size > 0, "at least 1 layer is needed"
        assert nn_input_dim == self.layers[0].input_dim, "nn_input_dim must be equal to that of first layer"
        for i in range(layer_size):
            if i > 0:
                assert self.layers[i].input_dim == self.layers[i - 1].output_dim, "input_dim of current layer must be equal to output_dim of last layer"
            if i < layer_size - 1:
                assert self.layers[i].act_func != ActFunc.Softmax, "softmax is only allowed as activation function on the last layer"

    def __sigint_handler(self, signum, frame):
        self.is_being_stoped = True
        logger.info('Training is being early stoped.')

    def train(self, max_epoch, learning_rate, batch_size, **kwargs):
        self.__validate(self.data_set.DIMENSIONS)
        assert learning_rate > 0, "learning rate must be greater than 0"
        assert max_epoch > 0, "max_epoch must be greater than 0"

        self.is_trained = False
        logger.info('Training started: max_epoch=%d, learning_rate=%f, batch_size=%d' % (max_epoch, learning_rate, batch_size))

        start_time = time.time()

        layer_size = len(self.layers)

        print_mean_square_error = kwargs['print_mean_square_error'] if 'print_mean_square_error' in kwargs else False
        print_cross_entropy = kwargs['print_cross_entropy'] if 'print_cross_entropy' in kwargs else False
        print_variance = kwargs['print_variance'] if 'print_variance' in kwargs else False

        total_cnt = max_epoch * self.data_set.TRAIN_SIZE
        process_bar = tqdm.tqdm(total=total_cnt, colour='cyan', ncols=120, unit_scale=True, desc="Training")

        for i in range(max_epoch):
            if self.is_being_stoped:
                break

            # Generate examples by batch randomly in each epoch.
            data_generator = self.data_set.data_generator(batch_size)

            for j, data in enumerate(data_generator):
                x_train = data[0]
                y_train = data[1]
                for k in range(layer_size):
                    if k == 0:
                        self.layers[k].calculate_forward(x_train)
                    else:
                        self.layers[k].calculate_forward(self.layers[k - 1].output)

                for k in reversed(range(layer_size)):
                    # Info of next layer.
                    next_layer = None if k == layer_size - 1 else self.layers[k + 1]
                    final_answer = y_train if k == layer_size - 1 else None

                    # Apply backward propogation.
                    self.layers[k].propagate_backward(next_layer, final_answer)
                
                for k in reversed(range(layer_size)):
                    # Info of last layer.
                    last_layer = None if k == 0 else self.layers[k - 1]
                    origin_input = x_train if k == 0 else None

                    # Update parameters.
                    self.layers[k].update_parameters(learning_rate * (1 - (i + 1) / max_epoch) + 5e-4, batch_size, last_layer, origin_input)

                current_cnt = i * self.data_set.TRAIN_SIZE + j * batch_size + len(data)
                process_bar.update(len(data[0]))

                if j % 100 == 0 or len(data[0]) != batch_size:
                    with numpy.printoptions(linewidth=numpy.inf):
                        used_time = time.time() - start_time

                        index = j * batch_size + len(data[0])
                        need_time = human_readable_time(used_time / current_cnt * total_cnt - used_time)
                        logger.debug("epoch={}, index={}, need_time={}".format(i + 1, index, need_time))

                        if print_mean_square_error:
                            mse = numpy.mean(numpy.square(self.layers[layer_size - 1].output - y_train))
                            logger.debug("-- mean_square_error={}".format(mse))

                        if print_cross_entropy:
                            cre = -numpy.mean(numpy.sum(y_train * numpy.log(self.layers[layer_size - 1].output), axis=1))
                            logger.debug("-- cross_entropy={}".format(cre))

                        if print_variance:
                            output_var = [layer.get_output_var() for layer in self.layers]
                            logger.debug("-- output_variance={}".format(output_var))

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

        for j in range(len(x_test)):
            for k in range(layer_size):
                if k == 0:
                    self.layers[k].calculate_forward(x_test[j])
                else:
                    self.layers[k].calculate_forward(self.layers[k - 1].output)
            ans = numpy.argmax(self.layers[layer_size - 1].output)
            std = numpy.argmax(y_test[j])
            if ans == std:
                cnt_correct += 1

        correct_rate = cnt_correct / len(y_test)
        return correct_rate

    def save_as_file(self, file_name=None):
        if not file_name:
            file_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + ".npz"
        file_path = os.path.join(prepare_directory(Env.MODEL_DIR), file_name)

        model = {}

        # Save layer size.
        model['layer_size'] = len(self.layers)

        # Save parameters of each layer.
        for i in range(len(self.layers)):
            model['layer_%d_weights' % (i + 1)] = self.layers[i].weights
            model['layer_%d_bias' % (i + 1)] = self.layers[i].bias
            model['layer_%d_act_func' % (i + 1)] = self.layers[i].act_func.value

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
            self.add_layer(Layer(weights.shape[0], weights.shape[1], act_func, weights, bias))

        self.is_trained = True
        logger.info('Parameters were loaded.')
