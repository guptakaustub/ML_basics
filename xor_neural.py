from numpy import *

class NeuralNet(object):

    def __init__(self, layerSize):
        self.iLayer = layerSize[0]
        self.hLayer = layerSize[1]
        self.oLayer = layerSize[2]

        # defining weights and biases for each layer
        self.W1 = random.randn(self.iLayer, self.hLayer)
        self.b1 = random.randn(1, self.hLayer)
        self.W2 = random.randn(self.hLayer, self.oLayer)
        self.b2 = random.randn(1, self.oLayer)

    def feedforward(self, X):
        self.z2 = dot(X, self.W1) + self.b1
        self.a2 = self.sigmoid(self.z2, deriv=False)
        self.z3 = dot(self.a2, self.W2) + self.b2
        a3 = self.sigmoid(self.z3, deriv=False)

        return a3

    def backpropagate(self, X, y):
        # declaring epoch for training
        epoch = 100000
        for i in range(epoch):
            print "Epoch: ", i+1
            # getting derivatives wrt to all weights in network
            # Matrix Equations of derivatives from Welch's Lab
            # dJ/dW2
            hx = self.feedforward(X)
            d3 = multiply(-(y - hx), self.sigmoid(self.z3,deriv=True))
            dw2 = dot(self.a2.T, d3)

            # dJ/dW1
            d2 = dot(d3, self.W2.T) * self.sigmoid(self.z2, deriv=True)
            dw1 = dot(X.T, d2)

            self.W1 -= 0.01 * dw1
            self.W2 -= 0.01 * dw2

    def sigmoid(self, z, deriv=False):
        if deriv is True:
            ez = 1 / (1 + exp(-z))
            return ez * (1 - ez)
        else:
            return 1 / (1 + exp(-z))


x_sample = array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_sample = array([
    [0],
    [1],
    [1],
    [0]
])

nn = NeuralNet([2,3,1])
# print nn.feedforward([0,1])
nn.backpropagate(x_sample,y_sample)
print nn.W1
print nn.W2
print nn.feedforward(x_sample)