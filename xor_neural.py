from numpy import *

class NeuralNet(object):

    def __init__(self, layerSize):
        self.iLayer = layerSize[0]
        self.hLayer = layerSize[1]
        self.oLayer = layerSize[2]

        # defining weights and biases for each layer
        self.W1 = random.randn(self.iLayer, self.hLayer)
        self.b1 = random.randn(self.hLayer)
        self.W2 = random.randn(self.hLayer, self.oLayer)
        self.b2 = random.randn(self.oLayer)

        # print "Printing layer counts: ",
        # print self.iLayer, self.hLayer, self.oLayer
        # print
        # print "W1: "
        # print self.W1
        # print "B1: "
        # print self.b1
        # print "W2: "
        # print self.W2
        # print "B2: "
        # print self.b2
        # print

    def feedforward(self, X):
        self.z2 = dot(X, self.W1) + self.b1
        self.a2 = self.sigmoid(self.z2, deriv=False)
        self.z3 = dot(self.a2, self.W2) + self.b2
        self.a3 = self.sigmoid(self.z3, deriv=False)

        # print "Z2: "
        # print z2
        # print "A2: "
        # print a2
        # print "Z3: "
        # print z3
        # print

        return self.a3

    def backpropagate(self, X, y):
        # getting derivatives wrt to all weights in network
        # dJ/dW2
        hx = self.feedforward(X)
        d3 = multiply(-(y - hx), self.sigmoid(self.z3,deriv=True))
        dw2 = dot(self.a2.T, d3)

        # dJ/dW1
        d2 = dot(d3, self.W2.T) * self.sigmoid(self.z2, deriv=True)
        dw1 = dot(X.T, d2)

        return dw1, dw2

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
print nn.feedforward([0,1])
a, b = nn.backpropagate(x_sample,y_sample)
print a
print b