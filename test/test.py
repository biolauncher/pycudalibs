import numpy
import _cunumpy
import _cublas

# create a vector
v=numpy.arange(1,100,dtype=numpy.float32)
V=_cunumpy.array_to_cuda(v)
print V

# create a matrix
a=numpy.array(numpy.random.rand(17, 20), dtype=numpy.float32)
A=_cunumpy.array_to_cuda(a)
print A

# create another matrix
b=numpy.array(numpy.random.rand(20, 50), dtype=numpy.float32)
B=_cunumpy.array_to_cuda(b)
print B

# create a third target matrix
c=numpy.zeros((17,50),dtype=numpy.float32)
C=_cunumpy.array_to_cuda(c)
print C

# do sgemm - as plain dot product

#_cublas.sgemm('n', 'n', 1.0, A, B, 0.0, C)

#print C

#r = _cunumpy.cuda_to_array(C)
#print r
