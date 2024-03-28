import tensorflow as tf
import numpy as np


"""
    This file is an implementation of a Neural Network like the Keras library.
    At the time the implementation is not finished.
"""

class Optimizers:
    """
        Implementation of several optimizers, such as:
            - SGD
            - SGD with momentum
            - Adagrad
            - RMSprop
    """
    class SGD:
        """
            Stochastic Gradient Descent Optimizer

            Parameters:
                        * learning_rate. float value.
        """
        def __init__(self, learning_rate=0.01):
            self.eta = learning_rate

        def get_gradients(self, dW, dB, *args):
            weight_gradient = self.eta * dW
            bias_gradient = self.eta * dB
            return weight_gradient, bias_gradient, 0.0, 0.0

    class MomentumSGD:
        """
            Stochastic Gradient Descent with Momentum
            Same as the SGD optimizer but adding Exponential Moving Average
            to the gradients.

            Parameters:
                        * learning_rate. float value.
                        * beta. float value, it's recommended to be between 0 and 1.
        """
        def __init__(self, learning_rate=0.01, beta=0.9):
            self.eta = learning_rate
            self.beta = beta

        def get_gradients(self, dW, dB, v_dW, v_dB):
            weight_gradient = v_dW * self.beta + self.eta * dW
            bias_gradient = v_dB * self.beta + self.eta * dB
            return weight_gradient, bias_gradient, weight_gradient, bias_gradient

    class Adagrad:
        """
            Adagrad Optimizer

            Parameters:
                        * learning_rate. float value.
                        * epsilon. float value.
        """
        def __init__(self, learning_rate=0.01, epsilon=1e-8):
            self.eta = learning_rate
            self.eps = epsilon

        def get_gradients(self, dW, dB, sumsq_dw, sumsq_db):
            sumsq_dw = sumsq_dw + tf.math.pow(dW, 2)
            sumsq_db = sumsq_db + tf.math.pow(dB, 2)
            weight_gradient = self.eta / (tf.math.sqrt(sumsq_dw + self.eps)) * dW
            bias_gradient = self.eta / (tf.math.sqrt(sumsq_db + self.eps)) * dB
            return weight_gradient, bias_gradient, sumsq_dw, sumsq_db

    class RMSprop:
        """
            RMSprop Optimizer

            Parameters:
                        * learning_rate. float value.
                        * beta. float value, it's recommended to be between 0 and 1.
                        * epsilon. float value.
        """
        def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
            self.eta = learning_rate
            self.beta = beta
            self.eps = epsilon

        def get_gradients(self, dW, dB, m_dw, m_db):
            m_dw = self.beta * m_dw + (1 - self.beta) * tf.math.pow(dW, 2)
            m_db = self.beta * m_db + (1 - self.beta) * tf.math.pow(dB, 2)
            weight_gradient = self.eta / (tf.math.sqrt(m_dw + self.eps)) * dW
            bias_gradient = self.eta / (tf.math.sqrt(m_db + self.eps)) * dB
            return weight_gradient, bias_gradient, m_dw, m_db



"""
    The following class is an implementation of a Neurons layer, being able to 
    specify the units, activation and input_dim.
"""
class Dense:
    """
        Dense layer

        Parameters:
                    * units. int value, number of neurons in the layer.
                    * activation. Activation function, available functions:
                                - 'relu'
                                - 'sigmoid'
                                - 'softmax'
                    * input_dim. Dimension of the input vector.
    """
    def __init__(self, units=10, activation='relu', input_dim=None):
        self._activation_functions = {'sigmoid': [self.Sigmoid, self.Sigmoid_prime],
                                      'relu': [self.ReLu, self.ReLu_prime],
                                      'softmax': [self.Softmax, self.Softmax_prime]}
        self.units = units
        self.activation, self.activation_derivative = self._activation_functions[activation]
        self.input_dim = input_dim
        if input_dim:
            self.weights = tf.Variable(tf.random.normal(shape=[units, input_dim],
                                                        mean=0.0, stddev=1.0, dtype=tf.float32))
            self.bias = tf.Variable(tf.zeros(shape=[units, 1], dtype=tf.float32))
            # Parameters for Optimizers
            self.v_dw = tf.Variable(tf.zeros(self.weights.shape, dtype=tf.float32))
            self.v_db = tf.Variable(tf.zeros(self.bias.shape, dtype=tf.float32))

    def init_parameters(self, input_dim):
        self.weights = tf.Variable(tf.random.normal(shape=[self.units, input_dim],
                                                    mean=0.0, stddev=1.0, dtype=tf.float32))
        self.bias = tf.zeros(shape=[self.units, 1], dtype=tf.float32)
        # Parameters for Optimizers
        self.v_dw = tf.Variable(tf.zeros(self.weights.shape, dtype=tf.float32))
        self.v_db = tf.Variable(tf.zeros(self.bias.shape, dtype=tf.float32))

    def update_parameters(self, dW, dB, optimizer):
        dw, db, v_dw, v_db = optimizer.get_gradients(dW, dB, self.v_dw, self.v_db)
        self.v_dw, self.v_db = v_dw, v_db
        self.weights = self.weights - dw
        self.bias = self.bias - db

    def dot_product(self, X):
        self.X = X  # (n_features, n_samples)
        self.A = self.activation(self.weights @ X + self.bias)
        return self.A

    def ReLu(self, Z):
        return tf.clip_by_value(Z, 0.0, float('inf'))

    def ReLu_prime(self, *args):
        return tf.cast(self.A > 0.0, dtype=tf.float32)

    def Softmax(self, Z):
        Z_T = tf.transpose(Z)
        Z_T = tf.exp(Z_T - tf.reduce_max(Z_T, axis=1, keepdims=True))
        A = Z_T / tf.reduce_sum(Z_T, axis=1, keepdims=True)
        return tf.transpose(A)

    def Softmax_prime(self, labels):
        return self.A - tf.transpose(labels)
        
    def Sigmoid(self, Z):
        A = 1 / (1 + tf.math.exp(-Z))
        return A

    def Sigmoid_prime(self, *args):
        return self.A * (1 - self.A)


"""
    Implementation of the Sequential model, where we can define our architecture of
    the Neural Network. Since I haven't define the compile method, we must specify the
    optimizer at
"""

class Sequential:
    """
        Sequential Model

        Parameters:
                    optimizer. An object from the Optimizers class.
    """
    def __init__(self, optimizer):
        self.architecture = []
        self.optimizer = optimizer

    def add(self, layer):
        if isinstance(layer, Dense):
            if layer.input_dim:
                self.architecture.append(layer)
            elif len(self.architecture):
                layer.init_parameters(self.architecture[-1].units)
                self.architecture.append(layer)
            else:
                raise ValueError(f'Parameter input_dim None value.')
        else:
            raise ValueError(f"Value must be Dense object.")

    def cross_entropy(self, y_true, y_pred):
        m = len(y_true)
        return -1 / m * tf.reduce_sum(tf.reduce_sum(tf.transpose(y_true) * tf.math.log(y_pred + 1e-10), axis=0))

    def predict(self, X):
        output = self.feedforward(X)
        return tf.argmax(output, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = tf.argmax(y, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), dtype=tf.float32))

    def feedforward(self, X):
        output = tf.transpose(X)  # (n_features, n_samples)
        for layer in self.architecture:
            output = layer.dot_product(output)
        return output

    def backpropagation(self, y, A):
        dL = tf.cast(1.0, dtype= tf.float32)
        m = len(y)
        for layer in self.architecture[::-1]:
            # Compute the gradients
            dZ = dL * layer.activation_derivative(y)
            dW = dZ @ tf.transpose(layer.X) * 1 / m
            dB = tf.reduce_sum(dZ, axis=1, keepdims=True) * 1 / m
            # Error for Next Layer
            dL = tf.transpose(layer.weights) @ dZ
            # Update parameters
            layer.update_parameters(dW, dB, self.optimizer)

    def fit(self, X, y, epochs, batch_size):
        x_const = tf.constant(X, dtype=tf.float32)
        y_const = tf.constant(y, dtype=tf.float32)
        x_tensor = tf.data.Dataset.from_tensor_slices(x_const)
        y_tensor = tf.data.Dataset.from_tensor_slices(y_const)
        samples = tf.data.Dataset.zip((x_tensor, y_tensor))

        for epoch in range(1, epochs + 1):
            batches = samples.shuffle(buffer_size=len(X)).batch(batch_size)
            for x_rand, y_rand in batches:
                output = self.feedforward(x_rand)
                self.backpropagation(y_rand, output)

            if epoch % 10 == 0:
                score = self.score(x_const, y_const)
                y_pred = self.feedforward(x_const)
                loss = self.cross_entropy(y_const, y_pred)
                print(f"Epoch: {epoch} -- Loss: {loss} -- Score: {score}")