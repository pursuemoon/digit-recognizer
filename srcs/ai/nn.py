# -- coding: utf-8 --

import os
import enum
import numpy
import tqdm
import time
import signal

from utils.log import logger
from utils.env import Env
from utils.time import human_readable_time
from utils.file import prepare_directory

from ai.calc import apply_activation, apply_activation_derivative, ActFunc


# Optimizers. For the hyperparameters used by different optimizers, see the implementations.

class OptType(enum.IntEnum):
    MiniBatch = 100,
    Momentum  = 200,
    AdaGrad   = 300,
    RmsProp   = 400,
    Adam      = 500,

class Optimizer(object):
    def __init__(self, opt_type=None, max_epoch=None, learning_rate=None, batch_size=None,
                 momentum_coef=None, rms_coef=None, epsilon=None, nd_array=None):
        if nd_array is None:
            assert isinstance(opt_type, OptType), "opt_type must be of enum type of OptType"

            assert max_epoch > 0, "max_epoch must be greater than 0"
            assert learning_rate > 0, "learning rate must be greater than 0"
            assert batch_size > 0, "batch_size must be set and greater than 0"

        self.opt_type = opt_type if nd_array is None else OptType(int(nd_array[0]))
        self.max_epoch = max_epoch if nd_array is None else int(nd_array[1])
        self.learning_rate = learning_rate if nd_array is None else nd_array[2]
        self.batch_size = batch_size if nd_array is None else nd_array[3]

        if self.opt_type == OptType.Momentum:
            if nd_array is None:
                assert 0 < momentum_coef < 1, "momentum_coef must be between 0 and 1"
            self.momentum_coef = momentum_coef if nd_array is None else nd_array[4]
        elif self.opt_type == OptType.AdaGrad:
            if nd_array is None:
                assert epsilon > 0, "epsilon must be greater than 0"
            self.epsilon = epsilon if nd_array is None else nd_array[4]
        elif self.opt_type == OptType.RmsProp:
            if nd_array is None:
                assert 0 < rms_coef < 1, "rms_coef must be between 0 and 1"
                assert epsilon > 0, "epsilon must be greater than 0"
            self.rms_coef = rms_coef if nd_array is None else nd_array[4]
            self.epsilon = epsilon if nd_array is None else nd_array[5]
        elif self.opt_type == OptType.Adam:
            if nd_array is None:
                assert 0 < momentum_coef < 1, "momentum_coef must be between 0 and 1"
                assert 0 < rms_coef < 1, "rms_coef must be between 0 and 1"
                assert epsilon > 0, "epsilon must be greater than 0"
            self.momentum_coef = momentum_coef if nd_array is None else nd_array[4]
            self.rms_coef = rms_coef if nd_array is None else nd_array[5]
            self.epsilon = epsilon if nd_array is None else nd_array[6]

    def mark_not_finish(self):
        self.max_epoch = -1

    def __str__(self):
        params_log = '[{}] max_epoch={}, learning_rate={}, batch_size={}'.format(self.opt_type.name, self.max_epoch, self.learning_rate, self.batch_size)

        if self.opt_type == OptType.Momentum:
            params_log += ', momentum_coef={}'.format(self.momentum_coef)
        elif self.opt_type == OptType.AdaGrad:
            params_log += ', epsilon={}'.format(self.epsilon)
        elif self.opt_type == OptType.RmsProp:
            params_log += ', rms_coef={}, epsilon={}'.format(self.rms_coef, self.epsilon)
        elif self.opt_type == OptType.Adam:
            params_log += ', momentum_coef={}, rms_coef={}, epsilon={}'.format(self.momentum_coef, self.rms_coef, self.epsilon)

        return params_log

    def as_nd_array(self):
        params = [self.opt_type, self.max_epoch, self.learning_rate, self.batch_size]
        if self.opt_type == OptType.Momentum:
            params.append(self.momentum_coef)
        elif self.opt_type == OptType.AdaGrad:
            params.append(self.epsilon)
        elif self.opt_type == OptType.RmsProp:
            params.append(self.rms_coef)
            params.append(self.epsilon)
        elif self.opt_type == OptType.Adam:
            params.append(self.momentum_coef)
            params.append(self.rms_coef)
            params.append(self.epsilon)
        return numpy.array(params, numpy.float64)

    def as_short_name(self):
        common_params = "{}E-{}s-{}-{}".format(self.max_epoch, self.batch_size, self.opt_type.name, self.learning_rate)

        special_params = ""
        if self.opt_type == OptType.Momentum:
            special_params = "{}m".format(self.momentum_coef)
        elif self.opt_type == OptType.AdaGrad:
            special_params = "{}e".format(self.epsilon)
        elif self.opt_type == OptType.RmsProp:
            special_params = "{}r-{}e".format(self.rms_coef, self.epsilon)
        elif self.opt_type == OptType.Adam:
            special_params = "{}m-{}r-{}e".format(self.momentum_coef, self.rms_coef, self.epsilon)

        name = "[{}]-[{}]".format(common_params, special_params)
        return name

# Layer of Neural Network.

class Layer(object):
    def __init__(self, input_dim, output_dim, act_func=None, weights=None, bias=None, random_seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if random_seed:
            numpy.random.seed(random_seed)

        if act_func is not None:
            assert isinstance(act_func, ActFunc), "act_func must be of enum type of ActFunc"
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

    def init_optimization_value(self, optimizer):
        if optimizer.opt_type == OptType.Momentum:
            self.momentum = numpy.zeros_like(self.weights)
        elif optimizer.opt_type == OptType.AdaGrad:
            self.descent_square_sum = numpy.zeros_like(self.weights)
        elif optimizer.opt_type == OptType.RmsProp:
            self.descent_square_sum = numpy.zeros_like(self.weights)
        elif optimizer.opt_type == OptType.Adam:
            self.momentum = numpy.zeros_like(self.weights)
            self.descent_square_sum = numpy.zeros_like(self.weights)

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
    
    def update_parameters(self, learning_rate, optimizer, last_layer, x_input):
        # Update parameters by back-propogation.
        layer_input = last_layer.output if last_layer else x_input

        gradient = numpy.zeros_like(self.weights)
        for idx in range(len(layer_input)):
            gradient += numpy.atleast_2d(layer_input[idx]).T * self.delta[idx]
        gradient /= len(layer_input)

        if optimizer.opt_type == OptType.Momentum:
            self.momentum = optimizer.momentum_coef * self.momentum + (1 - optimizer.momentum_coef) * gradient
            opt_learning_rate = learning_rate
            opt_gradient = self.momentum
        elif optimizer.opt_type == OptType.AdaGrad:
            self.descent_square_sum += numpy.square(gradient)
            opt_learning_rate = learning_rate / (numpy.sqrt(self.descent_square_sum) + optimizer.epsilon)
            opt_gradient = gradient
        elif optimizer.opt_type == OptType.RmsProp:
            self.descent_square_sum = optimizer.rms_coef * self.descent_square_sum + (1 - optimizer.rms_coef) * numpy.square(gradient)
            opt_learning_rate = learning_rate / (numpy.sqrt(self.descent_square_sum) + optimizer.epsilon)
            opt_gradient = gradient
        elif optimizer.opt_type == OptType.Adam:
            self.momentum = optimizer.momentum_coef * self.momentum + (1 - optimizer.momentum_coef) * gradient
            self.descent_square_sum = optimizer.rms_coef * self.descent_square_sum + (1 - optimizer.rms_coef) * numpy.square(gradient)
            opt_learning_rate = learning_rate / (numpy.sqrt(self.descent_square_sum) + optimizer.epsilon)
            opt_gradient = self.momentum
        else:
            opt_learning_rate = learning_rate
            opt_gradient = gradient

        self.weights -= opt_learning_rate * opt_gradient
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
        self.optimizers = []
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
        self.optimizers[-1].mark_not_finish()
        logger.info('Training is being early stoped.')

    def train(self, max_epoch, learning_rate, opt_type=OptType.MiniBatch,
              batch_size=20, momentum_coef=0.9, rms_coef=0.999, epsilon=1e-7,
              **kwargs):
        self.__validate(self.data_set.DIMENSIONS)

        # Initialize the current optimizer and its associated values.
        current_optimizer = Optimizer(opt_type, max_epoch, learning_rate, batch_size, momentum_coef, rms_coef, epsilon)
        self.optimizers.append(current_optimizer)
        for layer in self.layers:
            layer.init_optimization_value(current_optimizer)

        self.is_trained = False

        logger.info('Training started: {}'.format(current_optimizer))
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
                    self.layers[k].update_parameters(learning_rate, current_optimizer, last_layer, origin_input)

                current_cnt = i * self.data_set.TRAIN_SIZE + j * batch_size + len(data)
                process_bar.update(len(data[0]))

                if j % 100 == 0 or len(data[0]) != batch_size:
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

                        # if current_optimizer.mean_learning_rate:
                        #     extra_log += ", mean_learn_rate={}".format(current_optimizer.mean_learning_rate)

                        # if current_optimizer.mean_gradient:
                        #     extra_log += ", mean_gradient={}".format(current_optimizer.mean_gradient)

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

    def save_as_file(self, file_name=None, auto_name=False):
        if not file_name:
            if auto_name:
                pretrain_mark = 'Pre-' if len(self.optimizers) > 1 else ''
                structure = '-'.join(['{}-{}'.format(layer.output_dim, layer.act_func.name) for layer in self.layers])
                optimizer = self.optimizers[-1]
                file_name = '{}[{}]-{}.npz'.format(pretrain_mark, structure, optimizer.as_short_name())
            else:
                file_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + ".npz"

        file_path = os.path.join(prepare_directory(Env.MODEL_DIR), file_name)

        model = {}

        # Save parameters of each layer.
        model['layer_size'] = len(self.layers)
        for i in range(len(self.layers)):
            model['layer_%d_weights' % (i + 1)] = self.layers[i].weights
            model['layer_%d_bias' % (i + 1)] = self.layers[i].bias
            model['layer_%d_act_func' % (i + 1)] = self.layers[i].act_func.value

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
            self.add_layer(Layer(weights.shape[0], weights.shape[1], act_func, weights, bias))

        training_round = model['training_round']
        for i in range(training_round):
            opt_params = model['optimizer_at_training_round_%d' % (i + 1)]
            self.optimizers.append(Optimizer(nd_array=opt_params))
            logger.info('Pre-training was done: {}'.format(self.optimizers[-1]))

        self.is_trained = True
        logger.info('Parameters were loaded.')
