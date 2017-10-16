import numpy as np
from caispp import datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y)

def initialize_weights(d0, d1):
    return np.random.randn(d0, d1)


class NN(object):
    def __init__(self, layers):
        self.activations = []
        self.z = []
        self.weights = []
        self.layers = layers

        for layer in layers:
            self.activations.append(np.zeros(layer))
            self.z.append(np.zeros(layer))

        for i, layer in enumerate(layers[:-1]):
            self.weights.append(initialize_weights(layers[i + 1], layer))


    def feed_forward(self, inputs):
        #if len(inputs) != :
        #    raise ValueError('Network is not compatible with this many inputs')

        # a^0 = x
        self.activation_input = np.array(inputs[:])

        # to keep track of generalized a^i
        a_m = self.activation_input
        self.activations[0] = a_m
        self.z[0] = a_m

        for i, next_weight in enumerate(self.weights):
            # z^(m + 1) = W^(m + 1)a^m
            z_m_next = next_weight.dot(a_m)
            # a^(m + 1) = f(z^(m + 1))
            a_m_next = sigmoid(z_m_next)

            self.activations[i + 1] = a_m_next
            self.z[i + 1] = z_m_next

            a_m = a_m_next


    def back_propagate(self, targets, learning_rate):
        # s^L = f'(n^L) * (d J)/(d a)
        # (d J)/(d a) = a
        # In this case (d J)/(d a) = (a^L - t)

        error = self.activations[-1] - targets
        s_L= dsigmoid(self.activations[-1]) * error

        m = len(self.layers) - 2

        while m >= 0:
            if m != len(self.layers) - 2:
                # s^m = f'(z^m) * (W^(m+1))^T * s^(m+1)
                f_prime = dsigmoid(self.activations[m + 1])
                s_m = f_prime * self.weights[m + 1].T.dot(s_m_next)
            else:
                s_m = s_L

            # W^m (k+1) = W^m(k) - \alpha s^m * (a^(m-1))^T
            # Keep in mind k is fixed as this is a single iteration.
            self.weights[m] -= learning_rate * s_m.dot(self.activations[m].T)
            s_m_next = s_m
            m -= 1

        error = 0.0
        # Mean squared error.
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.activations[-1][k]) ** 2

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
X = X.reshape((X.shape[0], X.shape[1], 1))
Y = np.array(Y)
Y = Y.reshape((Y.shape[0], 1))

nn = NN([X.shape[1], 20, Y.shape[1]])
nn.train(list(zip(X, Y)))
