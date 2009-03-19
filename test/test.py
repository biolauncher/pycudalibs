import numpy
import _cunumpy
# create a vector
a=numpy.arange(1,100,dtype=numpy.float32)
print a.shape
print a
d=_cunumpy.array_to_cuda(a)
print d
# create a matrix
b=numpy.zeros((20,7),dtype=numpy.float32)
e=_cunumpy.array_to_cuda(b)
print b
print e
