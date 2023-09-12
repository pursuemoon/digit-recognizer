# -- coding: utf-8 --

import numpy
import enum

from utils.log import logger

from ai.calc import apply_activation, apply_activation_derivative, ActFunc, OptType

def pad_input(im, padding):
    '''
    Perform padding on an image.
    '''
    result = numpy.zeros((im.shape[0] +  2 * padding, im.shape[1] +  2 * padding))
    if padding > 0:
        result[padding:-padding, padding:-padding] = im
    else:
        result = im
    return result

def unfold_kernels(kernel_weights, input_shape, stride):
    '''
    Unfold origin kernels into a linear transformation matrix.
    '''
    assert len(input_shape) == 3
    assert len(kernel_weights.shape) == 4
    assert input_shape[0] == kernel_weights.shape[1], "channel number is expected to be equal among input and kernels"

    filter_num = int(kernel_weights.shape[0])
    channel_num = int(kernel_weights.shape[1])
    kernel_height = int(kernel_weights.shape[2])
    kernel_width = int(kernel_weights.shape[3])

    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    output_height = int((input_height - kernel_height) / stride + 1)
    output_width = int((input_width - kernel_width) / stride + 1)

    matrix_height = int(output_height * output_width)
    matrix_width = int(channel_num * input_height * input_width)

    kernel_matrices_t = numpy.zeros(shape=(filter_num, matrix_height, matrix_width))
    for f in range(filter_num):
        for c in range(channel_num):
            h = 0
            for i in range(output_height):
                for j in range(output_width):
                    w =  i * stride * input_width + j * stride + c * input_height * input_width
                    for p in range(kernel_height):
                        kernel_matrices_t[f][h][w:w+kernel_width] = kernel_weights[f][c][p]
                        w += input_width
                    h += 1
    return kernel_matrices_t

def conv_2d(x_2d, k_2d, stride):
    result = numpy.zeros(shape=(int((x_2d.shape[0]-k_2d.shape[0])/stride + 1), int((x_2d.shape[1]-k_2d.shape[1])/stride + 1)))
    rh = result.shape[0]
    rw = result.shape[1]
    for i in range(rh):
        for j in range(rw):
            result[i][j] = numpy.sum(x_2d[i*stride:i*stride+k_2d.shape[0], j*stride:j*stride+k_2d.shape[1]] * k_2d)
    return result

def conv_simple(x, kernels, stride):
    '''
    Perform convolution between an image and several kernels, using just simple approach.
    '''
    assert len(x.shape) == 3
    assert len(kernels.shape) == 4
    assert x.shape[0] == kernels.shape[1]

    filter_num = kernels.shape[0]
    channel_num = kernels.shape[1]

    result = numpy.zeros(shape=(filter_num, int((x.shape[1]-kernels.shape[1])/stride + 1), int((x.shape[2]-kernels.shape[3])/stride + 1)))
    for f in range(filter_num):
        for c in range(channel_num):
            result[f] += conv_2d(x[c], kernels[f][c], stride)
    return result

def conv_linearly(x, kernels, stride):
    '''
    Perform convolution between an image and several kernels, using linear transformation.
    '''
    assert len(x.shape) == 3
    assert len(kernels.shape) == 4
    assert x.shape[0] == kernels.shape[1]

    filter_num = kernels.shape[0]
    output_shape=(filter_num, int((x.shape[1]-kernels.shape[1])/stride + 1), int((x.shape[2]-kernels.shape[3])/stride + 1))

    kernel_matrices = numpy.array([k.T for k in unfold_kernels(kernels, x.shape, stride)])
    input_dim = numpy.prod(x.shape)
    result = numpy.dot(x.reshape(input_dim), kernel_matrices)
    result = numpy.reshape(result, newshape=output_shape)

    return result

def img2col(x, kernel_shape, stride):
    '''
    Unfold origin input with channels into a linear transformation matrix.
    '''
    assert len(x.shape) == 3
    assert len(kernel_shape) == 4
    assert x.shape[0] == kernel_shape[1]

    channel_num = x.shape[0]
    kernel_dim = kernel_shape[2] * kernel_shape[3]
    output_height = int((x.shape[1] - kernel_shape[2]) / stride + 1)
    output_width = int((x.shape[2] - kernel_shape[3]) / stride + 1)

    result = numpy.zeros(shape=(kernel_dim * channel_num, output_height * output_width))
    for i in range(output_height):
        for j in range(output_width):
            xi = i * stride
            xj = j * stride
            result[:, i*output_width+j] = numpy.reshape(x[:, xi:xi+kernel_shape[2], xj:xj+kernel_shape[3]], newshape=(kernel_dim * channel_num))
    
    return result


class LayerType(enum.IntEnum):
    Linear  = 1,
    Conv2d  = 2,
    Pooling = 3,

class AbstractLayer(object):
    def __init__(self, type=None, act_func=None):
        self.type = type
        self.act_func = act_func

        self.input_dim = None
        self.input = None

        self.output_dim = None
        self.output = None

        self.momentum = None
        self.descent_square_sum = None

    def calculate_forward(self, x):
        # input dimensions: (batch_size, input_dim)
        # output dimensions: (batch_size, output_dim)
        pass

    def propagate_backward(self, last_layer, y_output):
        pass
    
    def update_parameters(self, learning_rate, optimizer, step):
        pass

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

    def optimize(self, optimizer, learning_rate, gradient, step):
        if optimizer.opt_type == OptType.Momentum:
            self.momentum = optimizer.momentum_coef * self.momentum + (1 - optimizer.momentum_coef) * gradient
            opt_learning_rate = learning_rate
            opt_gradient = self.momentum
        elif optimizer.opt_type == OptType.AdaGrad:
            self.descent_square_sum += numpy.square(gradient)
            opt_learning_rate = learning_rate / numpy.sqrt(self.descent_square_sum + optimizer.epsilon)
            opt_gradient = gradient
        elif optimizer.opt_type == OptType.RmsProp:
            self.descent_square_sum = optimizer.rms_coef * self.descent_square_sum + (1 - optimizer.rms_coef) * numpy.square(gradient)
            opt_learning_rate = learning_rate / numpy.sqrt(self.descent_square_sum + optimizer.epsilon)
            opt_gradient = gradient
        elif optimizer.opt_type == OptType.Adam:
            self.momentum = optimizer.momentum_coef * self.momentum + (1 - optimizer.momentum_coef) * gradient
            self.descent_square_sum = optimizer.rms_coef * self.descent_square_sum + (1 - optimizer.rms_coef) * numpy.square(gradient)
            opt_learning_rate = learning_rate / numpy.sqrt(self.descent_square_sum / (1 - pow(optimizer.rms_coef, step)) + optimizer.epsilon)
            opt_gradient = self.momentum / (1 - pow(optimizer.momentum_coef, step))
        else:
            opt_learning_rate = learning_rate
            opt_gradient = gradient

        return opt_learning_rate, opt_gradient


# Linear Layer of Neural Network.

class LinearLayer(AbstractLayer):
    def __init__(self, input_dim, output_dim, act_func=None, weights=None, bias=None, random_seed=None):
        super().__init__(type=LayerType.Linear,act_func=act_func)
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

    def get_abstract(self):
        return 'Linear => weights.shape={}, act_func={}'.format(self.weights.shape, self.act_func.name)

    def calculate_forward(self, x):
        # Calculate outputs on each layer.
        self.input = x
        tmp = numpy.dot(x, self.weights) + self.bias
        self.output = apply_activation(tmp, self.act_func)

    def propagate_backward(self, last_layer, y_output):
        # Calculate the delta of the final layer.
        if y_output is not None:
            self.error = self.output - y_output
            self.delta = self.error * apply_activation_derivative(self.output, self.act_func)

        # Propagate the delta to the previous layer.
        last_layer.error = numpy.dot(self.delta, self.weights.T)
        last_layer.delta = last_layer.error * apply_activation_derivative(last_layer.output, last_layer.act_func)
    
    def update_parameters(self, learning_rate, optimizer, step):
        # Update parameters by back-propogation.
        batch_size = len(self.input)

        gradient = numpy.zeros_like(self.weights)
        for idx in range(batch_size):
            gradient += numpy.atleast_2d(self.input[idx]).T * self.delta[idx]
        gradient /= batch_size

        # Perform optimization.
        opt_learning_rate, opt_gradient = self.optimize(optimizer, learning_rate, gradient, step)

        # Gradient descent, using L2-Regularization.
        self.weights -= opt_learning_rate * (opt_gradient + optimizer.regular_coef / batch_size * self.weights)
        self.bias -= learning_rate * (numpy.mean(self.delta, axis=0) + optimizer.regular_coef / batch_size * self.bias)

    def get_output_std(self):
        return numpy.std(self.output)

    def get_output_var(self):
        return numpy.var(self.output)

# Convolutional Layer of Neural Network.

class Conv2dLayer(AbstractLayer):
    def __init__(self, input_shape, kernel_size, filter_num, stride, padding=0, 
                 act_func=None, weights=None, bias=None, random_seed=None):
        super().__init__(type=LayerType.Conv2d, act_func=act_func)
        # input_shape is a tuple of length 3: (input channels number, the image height, the image width).
        self.input_shape = input_shape
        self.input_dim = input_shape[0] * (input_shape[1] + 2 * padding) * (input_shape[2] + 2 * padding)

        self.kernel_size = kernel_size
        self.filter_num = filter_num

        self.stride = stride
        self.padding = padding

        output_height = int((input_shape[1] - kernel_size + 2 * padding) / stride + 1)
        output_width = int((input_shape[2] - kernel_size + 2 * padding) / stride + 1)

        self.output_shape = (filter_num, output_height, output_width)
        self.output_dim = numpy.prod(self.output_shape)

        if random_seed:
            numpy.random.seed(random_seed)

        if act_func is not None:
            assert isinstance(act_func, ActFunc), "act_func must be of enum type of ActFunc"
            self.act_func = act_func
        else:
            self.act_func = ActFunc.Identity

        channel_num = input_shape[0]
        filter_dim = channel_num * kernel_size ** 2

        # weights is an ndarray with 4 dimensions: (filter number, channel number, kernel height, kernel width).
        if weights is not None:
            self.weights = weights
        else:
            self.weights = numpy.random.normal(0, numpy.sqrt(1 / (filter_dim)), size=(filter_num, channel_num, kernel_size, kernel_size))

        # Kernel Matrix Transformation.
        # 1. kernel_matrices_t has 3 dimensions: (filter_num, output_dim, input_dim)
        # 2. kernel_matrices has 3 dimensions: (filter_num, input_dim, output_dim)
        self.pad_input_shape = (channel_num, input_shape[1]+2*padding, input_shape[2]+2*padding)
        self.kernel_matrices_t = unfold_kernels(self.weights, self.pad_input_shape, stride)
        self.kernel_matrices = numpy.array([kernel_t.T for kernel_t in self.kernel_matrices_t])

        if bias is not None:
            self.bias = bias
        else:
            self.bias = numpy.random.normal(0, numpy.sqrt(1 / (filter_dim)), size=filter_num)
        
        # Preprocess bias for later addition.
        self.bias_matrix = numpy.repeat(numpy.atleast_2d(self.bias).T, axis=1, repeats=output_height*output_width)

        self.output = None
        self.error = None
        self.delta = None

    def get_abstract(self):
        return 'Conv2d => input_shape={}, kernel_size={}, filter_num={}, act_func={}'.format(
            self.input_shape, self.kernel_size, self.filter_num, self.act_func.name)

    def calculate_forward(self, x):
        batch_size = x.shape[0]
        channel_num = self.input_shape[0]

        # Calculate padded height and padded width.
        p_height = self.input_shape[1] + 2 * self.padding
        p_width = self.input_shape[2] + 2 * self.padding
        p_dim = channel_num * p_height * p_width

        # Padding the inputs.
        if self.padding > 0:
            px = numpy.zeros(shape=(batch_size, channel_num, p_height, p_width))
            for i in range(batch_size):
                x_matrix = x[i].reshape(self.input_shape)
                for c in range(channel_num):
                    px[i][c] = pad_input(x_matrix[c], self.padding)
            px = numpy.reshape(px, newshape=(batch_size, p_dim))
        else:
            px = numpy.reshape(x, newshape=(batch_size, p_dim))

        self.input = px

        # Perform covolution by linear transformation.
        tmp = numpy.dot(px, self.kernel_matrices)

        # Add bias of each kernel.
        for i in range(batch_size):
            tmp[i] += self.bias_matrix

        tmp = apply_activation(tmp, self.act_func)
        self.output = numpy.reshape(tmp, newshape=(batch_size, self.output_dim))

    def propagate_backward(self, last_layer, y_output):
        # Since Conv2dLayer must not be the final layer, just propagate the delta to the previous layer.
        last_layer.error = numpy.dot(self.delta, self.kernel_matrices_t.reshape(self.output_dim, self.input_dim))
        last_layer.delta = last_layer.error * apply_activation_derivative(last_layer.output, last_layer.act_func)
    
    def update_parameters(self, learning_rate, optimizer, step):
        # Update parameters by back-propogation.
        batch_size = len(self.input)
        channel_num = self.input_shape[0]
        filter_num = self.output_shape[0]

        gradient = numpy.zeros_like(self.weights)
        for i in range(batch_size):
            in_img = numpy.reshape(self.input[i], newshape=self.pad_input_shape)
            linear_input = img2col(in_img, kernel_shape=self.weights.shape, stride=self.stride)

            f_delta = numpy.reshape(self.delta[i], newshape=(filter_num, numpy.prod(self.output_shape[1:])))
            gradient += numpy.dot(f_delta, linear_input.T).reshape(self.weights.shape)
        gradient /= batch_size

        # Perform optimization.
        opt_learning_rate, opt_gradient = self.optimize(optimizer, learning_rate, gradient, step)

        # Gradient descent, using L2-Regularization.
        self.weights -= opt_learning_rate * (opt_gradient + optimizer.regular_coef / batch_size * self.weights)

        bias_gradient = numpy.mean(numpy.sum(self.delta.reshape(batch_size, self.filter_num, numpy.prod(self.output_shape[1:])), axis=2), axis=0)
        self.bias -= learning_rate * (bias_gradient + optimizer.regular_coef / batch_size * self.bias)
        self.bias_matrix = numpy.repeat(numpy.atleast_2d(self.bias).T, axis=1, repeats=numpy.prod(self.output_shape[1:]))

        # Update the kernel matrix and its transposation.
        self.kernel_matrices_t = unfold_kernels(self.weights, self.pad_input_shape, self.stride)
        self.kernel_matrices = numpy.array([kernel_t.T for kernel_t in self.kernel_matrices_t])


if __name__ == '__main__':
    numpy.set_printoptions(linewidth=300)

    padding = 1
    stride = 2

    # One image with two channels.
    x = numpy.array([[[1,1,1],[1,1,1],[1,1,1]],[[0,0,0],[0,0,0],[0,0,0]]])
    px = numpy.zeros(shape=(len(x), x.shape[1] + padding * 2, x.shape[2] + padding * 2))
    for i in range(len(x)):
        px[i] = pad_input(x[i], padding)
    print('px=\n{}\n'.format(px))

    # One kernel.
    a = numpy.array([[[[1,2,3],[4,5,6],[7,8,9]],[[10,20,30],[40,50,60],[70,80,90]]]])
    km = numpy.array([k.T for k in unfold_kernels(a, (2,3+2*padding,3+2*padding), stride)])
    print('kernel=\n{}\n'.format(a))

    oh = int((x.shape[1] - a.shape[2] + padding * 2) / stride + 1)
    ow = int((x.shape[2] - a.shape[3] + padding * 2) / stride + 1)

    # Convolution method 1: kernel matrix transformation.
    y1 = numpy.dot(px.reshape((1, px.size)), km).reshape((len(a), oh, ow))

    print('y1=\n{}\n'.format(y1))

    # Convolution method 2: simple way.
    y2 = conv_simple(px, a, stride)
    print('y2=\n{}\n'.format(y2))

    # Convolution method 3: encapsulated function of kernel matrix transformation.
    y3 = conv_linearly(px, a, stride)
    print('y3=\n{}\n'.format(y3))

    # Convolution method 4: input matrix transformation.
    linear_px = img2col(px, kernel_shape=a.shape, stride=2)
    y4= numpy.dot(a.reshape(a.size), linear_px).reshape(len(a), oh, ow)
    print('y4=\n{}\n'.format(y4))