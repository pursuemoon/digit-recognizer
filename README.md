digit-recognizer
================

![license](https://badgen.net/badge/license/MIT/orange)

_A simple digit recognizer trained on MNIST dataset._

Prerequisite
------------

There are not many dependencies required to run this project, as it is more like a demo framework for learning machine learning. You only need python 3.0 and a few external modules in this project to build some simple models and complete training on the CPU. *Numpy* and some graphics libraries like *Matplotlib* are needed. Just follow th command below to complete the installation of external modules:

```bash
pip install -r requirements.txt
```

Quick Start
-----------

In order to better demonstrate the function of the framework, we use MNIST as the data set to build and implement a simple model with full connecting layers for training and testing.

```python
from ai.nn import Network
from ai.layer import LinearLayer
from ai.calc import ActFunc, OptType
from ai.data import MnistDataSet

# In order to get stable results, we generally set the random seed to a fixed value.
network = Network(data_set=MnistDataSet(), random_seed=1964)

# Two linear layers are added here. Strictly speaking, this artificial neural network has
# three layers of neurons: a 784-dimensional input layer, a 64-dimensional hidden layer, and
# a 10-dimensional output layer. The layers in the framework are based on the connections
# between neurons.
network.add_layer(LinearLayer(input_dim=1*28*28, output_dim=64, act_func=ActFunc.Relu))
network.add_layer(LinearLayer(input_dim=64, output_dim=10, act_func=ActFunc.Softmax))

# Train for just 1 epoch, using Mini-Batch Algorithm to optimize.
network.train(max_epoch=1, learning_rate=0.001, opt_type=OptType.MiniBatch, batch_size=10)

# Save the trained model.
network.save_as_file(auto_name=True)

# Test on test set of the specified data set.
network.test(show_mistakes=False)
```

Now we get our model which was trained on MNIST data set.

The model will be named according to its stucture and training parameters and will be saved in `models` directory. You could see more detail in the log file or console, like below. The test results show that the accuracy rate is 85.07%. Not great, but don't worry. We can optimize our training or even build a better model.

```bash
[nn.py:179] [INFO] Training ended. Time used: 00:00:07
[nn.py:186] [INFO] Test started: case_num=10000
[nn.py:228] [INFO] Test ended. Time used: 00:00:00
[nn.py:229] [INFO] correct_rate=0.850700
```

Training Optimization
---------------------

We can optimize the training by increasing the number of epochs, adjust learning rate, or using a more efficient way to optimze gradient descent. Now we use Adam Algorithm for optimization and train the above model for just 1 epoch. For setting of hyperparameters, you need to know some theory about optimizer.

```python
# Train for just 1 epoch, using Adam Algorithm to optimize.
network.train(max_epoch=1, learning_rate=0.001, regular_coef=0.001, opt_type=OptType.Adam,
              batch_size=10, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-8)
```

Now we get a much higher accuracy rate 95.62%. Not bad.

```bash
[nn.py:179] [INFO] Training ended. Time used: 00:00:19
[nn.py:186] [INFO] Test started: case_num=10000
[nn.py:228] [INFO] Test ended. Time used: 00:00:00
[nn.py:229] [INFO] correct_rate=0.956200
```

The framework provides several optimization algorithms to choose from, like Mini-Batch, Momentum, Ada-Grad, RMS-Prop, Adam. Pick one you like.

Network Building
----------------

You can also build your network using more complex layer, like convolutional layer and so on. In general, more complex models require more time to train, and some methods must be used to avoid overfitting. Now we build a network with 1 Conv2DLayer, 1 DropoutLayer, 3 LinearLayer, and train it for more epochs.

```python
from ai.nn import Network
from ai.layer import LinearLayer, Conv2dLayer, DropoutLayer
from ai.calc import ActFunc, OptType
from ai.data import MnistDataSet

network = Network(data_set=MnistDataSet())

network.add_layer(Conv2dLayer(input_shape=(1,28,28), kernel_size=5, filter_num=3, stride=3, padding=1, act_func=ActFunc.Relu))
network.add_layer(LinearLayer(input_dim=3*9*9, output_dim=64, act_func=ActFunc.Relu))
network.add_layer(LinearLayer(input_dim=64, output_dim=64, act_func=ActFunc.Relu))
network.add_layer(DropoutLayer(dropout_prob=0.1))
network.add_layer(LinearLayer(input_dim=64, output_dim=10, act_func=ActFunc.Softmax))

network.train(max_epoch=10, learning_rate=0.001, regular_coef=0.001, opt_type=OptType.Adam, 
              batch_size=10, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-8)

network.save_as_file(auto_name=True)
network.test(show_mistakes=False)
```

Now we get a higher accuracy rate 98.01%. A pretty good improvement!

Remember we save models we have trained? We can also load them and train them for more epochs, with some other new optimizer, or even use other data set. In `models` directory, this framework provide some pre-trained model. You can load and use them very easily, without a long time training.

Here show a example for loading a CNN model named `mnist-cnn.npz` to test on MNIST.

```python
network = Network(data_set=MnistDataSet())
network.load_from_file('mnist-cnn.npz')
network.test(show_mistakes=False)
```

```bash
[nn.py:29] [INFO] Seed was set to Network: random_seed=1964
[nn.py:33] [INFO] Adding layer-1: Conv2d => input_shape=(1, 28, 28), kernel_size=5, filter_num=10, stride=2, padding=1, act_func=Relu
[nn.py:33] [INFO] Adding layer-2: Conv2d => input_shape=(10, 13, 13), kernel_size=3, filter_num=20, stride=2, padding=0, act_func=Relu
[nn.py:33] [INFO] Adding layer-3: Linear => input_dim=720, output_dim=256, act_func=Relu
[nn.py:33] [INFO] Adding layer-4: Linear => input_dim=256, output_dim=64, act_func=Relu
[nn.py:33] [INFO] Adding layer-5: Linear => input_dim=64, output_dim=10, act_func=Softmax
[nn.py:345] [INFO] Pre-trained: [mnist] [Adam] max_epoch=50, learning_rate=0.001, batch_size=10, regular_coef=0.001, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-08
[nn.py:345] [INFO] Pre-trained: [kaggle-mnist] [Adam] max_epoch=2, learning_rate=0.001, batch_size=10, regular_coef=0.001, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-08
[nn.py:345] [INFO] Pre-trained: [kaggle-mnist] [Adam] max_epoch=10, learning_rate=0.001, batch_size=10, regular_coef=0.001, momentum_coef=0.99, rms_coef=0.999, epsilon=1e-08
[nn.py:348] [INFO] Parameters were loaded.
[nn.py:186] [INFO] Test started: case_num=10000
[nn.py:228] [INFO] Test ended. Time used: 00:00:13
[nn.py:229] [INFO] correct_rate=0.993700
```

Data Set and Pretrained Model
-----------------------------

Now the project has 2 data set, mnist and kaggle-mnist. You could see them in the `ai/data` directory. More data sets and pretrained models will be import in the future, maybe or not. ;)

Maintainer
----------

[@pursuemoon](https://github.com/pursuemoon)


License
-------

[MIT](LICENSE) © pursuemoon.