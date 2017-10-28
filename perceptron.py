from numpy import *
import matplotlib.pyplot as plt

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,3.242068655,1],
	[7.673756466,3.508563011,1]]

def sigmoid(z):
    return 1/(1 + exp(-z))

def error(w1, w2, b):
    e = 0
    for points in dataset:
        e += (points[2] - sigmoid(w1*points[0] + w2*points[1] + b)) ** 2
    return e

def gradient_descent(w1, w2, b, learningRate):
    cost_w1 = 0
    cost_w2 = 0
    cost_b = 0

    for points in dataset:
        # Weighted Sum
        w_sum = (points[0] * w1) + (points[1] * w2 + b)
        # Cost Function
        cost = (sigmoid(w_sum) - points[2]) ** 2
        # Getting Partial Derivative of cost function w.r.t w1, w2, b
        # d(cost)/d(hypothesis) : hypothesis = 1/1 + sig(z)
        cost_h = 2 * (sigmoid(w_sum) - points[2])
        # d(hypothesis)/d(z)
        h_z = sigmoid(w_sum) * (1 - sigmoid(w_sum))
        # d(z)/d(w1)
        z_w1 = points[0]
        # d(z)/d(w1)
        z_w2 = points[1]
        # d(z)/d(b) = 1

        # d(cost)/d(w1)
        cost_w1 += cost_h * h_z * z_w1
        # d(cost)/d(w2)
        cost_w2 += cost_h * h_z * z_w2
        # d(cost)/d(b)
        cost_b += cost_h * h_z * 1

    new_w1 = w1 - (learningRate * cost_w1) / len(dataset)
    new_w2 = w2 - (learningRate * cost_w2) / len(dataset)
    new_b = b - (learningRate * cost_b) / len(dataset)
    return new_w1, new_w2, new_b


def trainPerceptron():
    # Defining Hyper-parameters
    init_w1 = 0
    init_w2 = 0
    init_b = 0
    learningR = 0.01
    epochs = 100000
    for i in range(epochs):
        init_w1, init_w2, init_b = gradient_descent(init_w1, init_w2, init_b, learningR)
        print init_w1, init_w2, init_b
    return init_w1, init_w2, init_b

def makePrediction(a, b, c, x1, x2):
    return round(sigmoid(a * x1 + b * x2 + c))

ei = error(0,0,0)
#a, b, c = trainPerceptron()
ef = error(2.08636250718, -3.06541386122, -1.21915138792)
print "Accuracy: {0}".format(1-ef/ei)
print "Prediction"
print makePrediction(2.08636250718, -3.06541386122, -1.21915138792, 8.675418651, 3.242068655)
