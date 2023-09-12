# -- coding: utf-8 --

import numpy
import enum
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

# Activation Functions.

class ActFunc(enum.IntEnum):
    Identity  = 100,
    Sigmoid   = 200,
    Relu      = 300,
    LeakyRelu = 301,
    Tanh      = 400,
    Softmax   = 500,

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

def leaky_relu(x):
    return numpy.maximum(0.01 * x, x)

def leaky_relu_derivative(activated):
    tmp = numpy.copy(activated)
    tmp[activated > 0] = 1
    tmp[activated <= 0] = 0.01
    return tmp

def tanh(x):
    # Original formula: tanh(x) = (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x)).
    return numpy.tanh(x)

def tanh_derivative(activated):
    return 1 - numpy.square(activated)

def softmax(x):
    # Original formula: softmax(x) = numpy.exp(x) / numpy.sum(numpy.exp(x), axis=column_axis, keepdims=True).
    if isinstance(x, numpy.ndarray):
        column_axis = 0 if len(x.shape) == 1 else 1
        e_x = numpy.exp(x - numpy.max(x, axis=column_axis, keepdims=True))
        return e_x / numpy.sum(e_x, axis=column_axis, keepdims=True)
    else:
        # For x is just a number.
        return 1

def apply_activation(x, act_func):
    if act_func == ActFunc.Identity:
        return identity(x)
    if act_func == ActFunc.Sigmoid:
        return sigmoid(x)
    if act_func == ActFunc.Relu:
        return relu(x)
    if act_func == ActFunc.LeakyRelu:
        return leaky_relu(x)
    if act_func == ActFunc.Tanh:
        return tanh(x)
    if act_func == ActFunc.Softmax:
        return softmax(x)

def apply_activation_derivative(x, act_func):
    if act_func == ActFunc.Identity:
        return identity_derivative(x)
    if act_func == ActFunc.Sigmoid:
        return sigmoid_derivative(x)
    if act_func == ActFunc.Relu:
        return relu_derivative(x)
    if act_func == ActFunc.LeakyRelu:
        return leaky_relu_derivative(x)
    if act_func == ActFunc.Tanh:
        return tanh_derivative(x)
    if act_func == ActFunc.Softmax:
        # Since the softmax function is not a univariate function, i.e. its output depends on all
        # dimensions of the x vector, the derivative of the softmax function can not be represented
        # by activated values themselves. Fortunately, the formula which calculates the partial
        # derivative of the loss function can be simplified, so 1 is returned here.
        #
        # Note that the softmax function is only allowed as an activation function in the output
        # layer to transform logits into probability distribution.
        return 1


# Optimizers. For the hyperparameters used by different optimizers, see the implementations.

class OptType(enum.IntEnum):
    MiniBatch = 100,
    Momentum  = 200,
    AdaGrad   = 300,
    RmsProp   = 400,
    Adam      = 500,


# Plotting graphs of different activation functions.

if __name__ == "__main__":
    x = numpy.arange(-2, 2, 0.001)

    y_identity = identity(x)
    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    y_leaky_relu = leaky_relu(x)
    y_tanh = tanh(x)

    fig, ax = plt.subplots()

    ax.plot(x, y_identity, label='identity', color='#ff77aa', linestyle='-')
    ax.plot(x, identity_derivative(y_identity), label='identity\'', color='#ff77aa', linestyle='--', visible=False, linewidth=1)

    ax.plot(x, y_sigmoid, label='sigmoid', color='#555533', linestyle='-')
    ax.plot(x, sigmoid_derivative(y_sigmoid), label='sigmoid\'', color='#555533', linestyle='--', visible=False, linewidth=1)

    ax.plot(x, y_relu, label='relu', color='#cc9955', linestyle='-')
    ax.plot(x, relu_derivative(y_relu), label='relu\'', color='#cc9955', linestyle='--', visible=False, linewidth=1)

    ax.plot(x, y_leaky_relu, label='leaky relu', color='#dd66aa', linestyle='-')
    ax.plot(x, leaky_relu_derivative(y_leaky_relu), label='leaky relu\'', color='#dd66aa', linestyle='--', visible=False, linewidth=1)

    ax.plot(x, y_tanh, label='tanh', color='#33cc44', linestyle='-')
    ax.plot(x, tanh_derivative(y_tanh), label='tanh\'', color='#33cc44', linestyle='--', visible=False, linewidth=1)

    legend = ax.legend(loc='lower right')

    lines = ax.get_lines()
    labels = [line.get_label() for line in lines]

    plt.ylim(-2 - 0.1, 2 + 0.1)
    plt.subplots_adjust(left=0.3)

    rax = plt.axes([0.05, 0.2, 0.15, 0.03 * len(lines)])

    def show_function(label):
        line = lines[labels.index(label)]
        line.set_visible(not line.get_visible())
        plt.draw()

    check = widgets.CheckButtons(rax, labels=labels, actives=[True if i % 2 == 0 else False for i in range(len(labels))])
    check.on_clicked(show_function)

    plt.show()