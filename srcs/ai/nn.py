# -- coding: utf-8 --

import os
import enum
import numpy
import time
import signal

from utils.log import logger
from utils.env import Env
from utils.time import human_readable_time
from utils.file import prepare_directory

# Activation Functions

class ActFunc(enum.IntEnum):
    Identity = 1,
    Sigmoid = 2,
    Relu = 3,
    Tanh = 4,

def identity(x):
    return x

def identity_derivative(activated):
    return numpy.ones_like(activated)

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(activated):
    return activated * (1 - activated)

def relu(x):
    return numpy.maximum(0, x)

def relu_derivative(activated):
    tmp = numpy.copy(activated)
    tmp[activated > 0] = 1
    tmp[activated <= 0] = 0
    return tmp

def tanh(x):
    return (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))

def tanh_derivative(activated):
    return 1 - numpy.square(activated)

def apply_activation(x, act_func):
    if act_func == ActFunc.Identity:
        return identity(x)
    if act_func == ActFunc.Sigmoid:
        return sigmoid(x)
    if act_func == ActFunc.Relu:
        return relu(x)
    if act_func == ActFunc.Tanh:
        return tanh(x)

def apply_activation_derivative(x, act_func):
    if act_func == ActFunc.Identity:
        return identity_derivative(x)
    if act_func == ActFunc.Sigmoid:
        return sigmoid_derivative(x)
    if act_func == ActFunc.Relu:
        return relu_derivative(x)
    if act_func == ActFunc.Tanh:
        return tanh_derivative(x)

# Layer of Neural Network.

class Layer(object):
    def __init__(self, input_dim, output_dim, act_func=None, weights=None, bias=None, random_seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if random_seed:
            numpy.random.seed(random_seed)

        if weights is not None:
            assert input_dim == weights.shape[0], "weights.shape[0] must be equal to input_dim"
            assert output_dim == weights.shape[1], "weights.shape[1] must be equal to output_dim"
            self.weights = weights
        else:
            # 1. Self Defined Initialization
            # self.weights = numpy.random.randn(input_dim, output_dim) / numpy.sqrt(input_dim + output_dim)

            # 2. Xavier Initialization I.
            # limit = numpy.sqrt(6 / (input_dim + output_dim))
            # self.weights = numpy.random.uniform(-limit, limit, size=(input_dim, output_dim))

            # 3. Xavier Initializatioin II.
            self.weights = numpy.random.normal(0, numpy.sqrt(2 / (input_dim + output_dim)), size=(input_dim, output_dim))

        if bias is not None:
            assert output_dim == bias.shape[0], "bias.shape[0] must be equal to output_dim"
            self.bias = bias
        else:
            # 1. Zero Initialization
            # self.bias = numpy.zeros((output_dim), dtype=numpy.float64) 

            # 2. Xavier Initializatioin I.
            # limit = numpy.sqrt(6 / (input_dim + output_dim))
            # self.bias = numpy.random.uniform(-limit, limit, size=output_dim)

            # 3. Xavier Initializatioin II.
            self.bias = numpy.random.normal(0, numpy.sqrt(2 / (input_dim + output_dim)), size=output_dim)

        if act_func is not None:
            assert isinstance(act_func, ActFunc), "act_func must be of enumeration type of ActFunc"
            self.act_func = act_func
        else:
            self.act_func = ActFunc.Identity

        self.output = None
        self.error = None
        self.delta = None

    def calculate_forward(self, x):
        # Calcuate outputs on each layer.
        tmp = numpy.dot(x, self.weights) + self.bias
        self.output = apply_activation(tmp, self.act_func)

    def propagate_backward(self, next_layer, y_onehot):
        # Calculate errors from the next layer.
        if not next_layer:
            self.error = self.output - y_onehot
        else:
            self.error = numpy.dot(next_layer.weights, next_layer.delta)

        # Calculate deltas of this layer.
        self.delta = self.error * apply_activation_derivative(self.output, self.act_func)
    
    def update_parameters(self, learning_rate, last_layer, x_input):
        # Update parameters by back-propogation.
        layer_input = last_layer.output if last_layer else x_input
        self.weights -= learning_rate * numpy.atleast_2d(layer_input).T * self.delta
        self.bias -= learning_rate * self.delta

    def get_weights_std(self):
        return numpy.std(self.weights)

    def get_weights_var(self):
        return numpy.var(self.weights)

    def get_bias_std(self):
        return numpy.std(self.bias)

    def get_bias_var(self):
        return numpy.var(self.bias)


class Network(object):
    def __init__(self, random_seed=None):
        self.layers = []
        self.is_trained = False
        self.is_being_stoped = False
        if random_seed:
            numpy.random.seed(random_seed)
            logger.info('Seed was set: random_seed=%d' % random_seed)
        signal.signal(signal.SIGINT, self.__sigint_handler)

    def add_layer(self, layer):
        logger.info('Adding layer-{}: weights.shape={}'.format(len(self.layers) + 1, layer.weights.shape))
        self.layers.append(layer)

    def __validate(self, nn_input_dim):
        layer_size = len(self.layers)
        assert layer_size > 0, "at least 1 layer is needed"
        assert nn_input_dim == self.layers[0].input_dim, "nn_input_dim must be equal to that of first layer"
        for i in range(layer_size):
            if i > 0:
                assert self.layers[i].input_dim == self.layers[i - 1].output_dim, "input_dim of current layer must be equal to output_dim of last layer"

    def __sigint_handler(self, signum, frame):
        self.is_being_stoped = True
        logger.info('Training is being early stoped.')

    def train(self, x_train, y_train, learning_rate, max_epoch, **kwargs):
        self.__validate(x_train.shape[1])
        assert len(x_train) == len(y_train), "length of y_train must be equal to length of x_train"
        assert learning_rate > 0, "learning rate must be greater than 0"
        assert max_epoch > 0, "max_epoch must be greater than 0"

        self.is_trained = False
        logger.info('Training started.')

        start_time = time.time()

        layer_size = len(self.layers)

        y_onehot = numpy.zeros((len(x_train), self.layers[layer_size - 1].output_dim), numpy.float64)
        y_onehot[numpy.arange(len(y_train)), y_train] = 1

        print_mod = kwargs['print_mod'] if 'print_mod' in kwargs else 5000
        print_mse = kwargs['print_mse'] if 'print_mse' in kwargs else False
        print_var = kwargs['print_var'] if 'print_var' in kwargs else False

        for i in range(max_epoch):
            if self.is_being_stoped:
                break
            for j in range(len(x_train)):
                for k in range(layer_size):
                    if k == 0:
                        self.layers[k].calculate_forward(x_train[j])
                    else:
                        self.layers[k].calculate_forward(self.layers[k - 1].output)

                for k in reversed(range(layer_size)):
                    # Info of next layer.
                    next_layer = None if k == layer_size - 1 else self.layers[k + 1]
                    final_answer = y_onehot[j] if k == layer_size - 1 else None

                    # Apply backward propogation.
                    self.layers[k].propagate_backward(next_layer, final_answer)
                
                for k in reversed(range(layer_size)):
                    # Info of last layer.
                    last_layer = None if k == 0 else self.layers[k - 1]
                    origin_input = x_train[j] if k == 0 else None

                    # Update parameters.
                    self.layers[k].update_parameters(learning_rate, last_layer, origin_input)

                if print_mod > 0 and j % print_mod == (print_mod - 1):
                    with numpy.printoptions(linewidth=numpy.inf):
                        total_cnt = max_epoch * len(x_train)
                        current_cnt = i * len(x_train) + (j + 1)
                        used_time = time.time() - start_time
                        need_time = used_time / current_cnt * total_cnt - used_time
                        logger.debug("epoch={}, index={}, ct={}".format(i + 1, j + 1, human_readable_time(need_time)))

                        if print_mse:
                            mse = numpy.mean(numpy.square(self.layers[layer_size - 1].output - y_onehot[j]))
                            logger.debug("-- mse={}".format(mse))

                        if print_var:
                            weights_var = [layer.get_weights_var() for layer in self.layers]
                            bias_var = [layer.get_bias_var() for layer in self.layers]
                            logger.debug("-- weights_var={}, bias_var={}".format(weights_var, bias_var))

                if self.is_being_stoped:
                    break

        self.is_trained = True
        
        end_time = time.time()
        logger.info('Training ended. Time used: {}'.format(human_readable_time(end_time - start_time)))

    def test(self, x_test, y_test):
        assert self.is_trained, "neural network must be trained first before using it"

        cnt_correct = 0
        layer_size = len(self.layers)

        for j in range(len(x_test)):
            for k in range(layer_size):
                if k == 0:
                    self.layers[k].calculate_forward(x_test[j])
                else:
                    self.layers[k].calculate_forward(self.layers[k - 1].output)
            ans = numpy.argmax(self.layers[layer_size - 1].output)
            if ans == y_test[j]:
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
