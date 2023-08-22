# -- coding: utf-8 --

import numpy
import enum
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

# Activation Functions.

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
    # Original formula: tanh(x) = (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x)).
    return numpy.tanh(x)

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


# Plotting graphs of different activation functions.

if __name__ == "__main__":
    x = numpy.arange(-3, 3, 0.05)

    y_identity = identity(x)
    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    y_tanh = tanh(x)

    fig, ax = plt.subplots()

    ax.plot(x, y_identity, label='identity', color='#ff77aa', linestyle='-')
    ax.plot(x, identity_derivative(y_identity), label='identity\'', color='#ff77aa', linestyle='--', visible=False)

    ax.plot(x, y_sigmoid, label='sigmoid', color='#555533', linestyle='-')
    ax.plot(x, sigmoid_derivative(y_sigmoid), label='sigmoid\'', color='#555533', linestyle='--', visible=False)

    ax.plot(x, y_relu, label='relu', color='#cc9955', linestyle='-')
    ax.plot(x, relu_derivative(y_relu), label='relu\'', color='#cc9955', linestyle='--', visible=False)

    ax.plot(x, y_tanh, label='tanh', color='#33cc44', linestyle='-')
    ax.plot(x, tanh_derivative(y_tanh), label='tanh\'', color='#33cc44', linestyle='--', visible=False)

    legend = ax.legend(loc='lower right')

    lines = ax.get_lines()
    labels = [line.get_label() for line in lines]

    plt.ylim(-1 - 0.1, 1 + 0.1)
    plt.subplots_adjust(left=0.3)

    rax = plt.axes([0.01, 0.2, 0.15, 0.03 * len(lines)])

    def show_function(label):
        line = lines[labels.index(label)]
        line.set_visible(not line.get_visible())
        plt.draw()

    check = widgets.CheckButtons(rax, labels=labels, actives=[True if i % 2 == 0 else False for i in range(len(labels))])
    check.on_clicked(show_function)

    plt.show()