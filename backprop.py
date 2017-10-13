import numpy as np
from caispp import datasets

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y)

def initialize_weights(dim):
    return np.random.randn(dim)


class NN(object):
    def __init__(self, layers):
        self.activations = []
        self.weights = []
        self.layers = layers

        for layer in layers:
            self.activations.append(np.zeros(layers))

        for i, layer in enumerate(layers[:-1]):
            self.weights.append(initialize_weights(layer, layers[i + 1]))


    def feed_forward(self, inputs):
        if len(inputs) != self.input:
            raise ValueError('Network is not compatible with this many inputs')

        # a^0 = x
        self.activation_input = np.array(inputs[:])

        ai = np.array(inputs[:])
        self.activations[0] = ai

        for i, weight_hidden in enumerate(self.weights_hiddens):
            # n^(m + 1) = W^(m + 1)a^m
            ni = weight_hidden.dot(ai)
            # a^(m + 1) = f(n^(m + 1))
            ai = sigmoid(ni)
            self.activations[i + 1] = ai


    def back_propagate(self, targets, learning_rate):
        if len(targets) != self.output:
            raise ValueError('Number of targets not correct')

        # s^L = f'(n^L) * (d J)/(d a)
        # (d J)/(d a) = a
        # In this case (d J)/(d a) = (a^L - t)

        error = self.activation_output - targets
        sL= dsigmoid(self.activations[-1]) * error

        # as we have already computed the last.
        s_next = sL
        i = len(layers) - 2
        while i >= 1:
            si = dsigmoid(self.activations[i]).dot(self.weights[i + 1].T).dot(s_next)
            self.weights[i] -= learning_rate * si.dot(self.activations[i - 1].T)
            s_next = si

        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.activation[-1][k]) ** 2

        return error


    def train(self, patterns, iterations = 3000, learning_rate = 0.0002):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]

                self.feed_forward(inputs)
                error = self.back_propagate(targets, learning_rate)

            print('%i: Error %.5f' % (i, error))

X, Y = datasets.download_uci_seeds()

# Load the data into numpy arrays.
X = np.array(X)
Y = np.array(Y)
Y = Y.reshape((Y.shape[0], 1))

nn = NN([X.shape[1], 20, Y.shape[1]])
nn.train(list(zip(X, Y)))
