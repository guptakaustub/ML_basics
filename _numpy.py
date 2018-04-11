from numpy import *

# def sigmoid(z):
#     print 1/(1+exp(-z))
#
# sigmoid(2)
#
# print array([i for i in range(5)], ndmin=3, dtype=complex)
# print "Initialization and Datatype"
# a = array([[x for x in range(7)],[x for x in range(7)],[x for x in range(7)],[x for x in range(7)]])
# print a, a.shape
# print "Checking Shape"
# a.shape = (7,4)
# print a, a.shape
# print "Changing Shape"
# print a[:3]
# print "Slicing 1"
# print a[4:]
# print "Slicing 2"
# a = a.reshape(4,7)
# print a, a.shape
# """ use RESHAPE instead of SHAPE """
# print "Sahping with RESHAPE"
# b = arange(10)
# print b, b.ndim
# print "Dimension"
# b = b.reshape(2,5)
# print b, b.shape, b.ndim
# print "Shape and Dimension"
# c = zeros(9)
# print c
# print  c.reshape(3,3)
# print "Zeros"
# d = ones(4)
# print d.reshape(2,2)
# print "Ones"
# print ones((5,5))
# print "Ones with extra dimension"
# print arange(10,101,15)
# print "Arange"
# print linspace(1,2,5)
# print "Linspace"
# print
# n1 = arange(1,10,1).reshape(3,3)
# n2 = arange(11,14)
# print n1, n1.shape
# print n2, n2.shape
# print "Add"
# print n1 + n2
# print "Product"
# print n1 * n2
# print "Dot"
# print dot(n1,n2), dot(n1,n2).shape
# print
# print sigmoid(n1)
# print random.random((3,2))
# print
# print random.randn(1,2)
# x = random.randn(3,1)
# cross-entropy
# print x
# print exp(x)
# print sum(exp(x))
# print exp(x[2])/sum(exp(x))
# print -log(exp(x[2])/sum(exp(x)))
for i in unpackbits(array([range(256)], dtype=uint8).T, axis=1):
    print i