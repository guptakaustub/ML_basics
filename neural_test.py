from numpy import *

def sigmoid(z, deriv=False):
    if deriv is True:
        ez = 1/(1+exp(-z))
        return ez * (1 - ez)
    else:
        return 1/(1+exp(-z))
#
# inp = random.rand(3)
# W = random.rand(3,2)
# b = random.rand(3)
#
print sigmoid(19.8, False)

# sh = (2,2,1)
# Wh = []
# for (x,z) in zip(sh[:-1],sh[1:]):
#     Wh.append(random.normal(scale=0.1, size=(z,x+1)))
# print Wh[0][0]
# print Wh[0][1]
# print
# print Wh[1][0]

# print vstack([1,2,3])
# print ones([1,5,6])