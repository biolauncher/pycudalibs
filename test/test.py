import numpy
import _cunumpy
import _cublas


_cublas.init()

# create a vector
v = numpy.arange(1,100,dtype=numpy.float32)
V = _cunumpy.array_to_cuda(v)
print "V=", V

# create a matrix
a = numpy.array(numpy.random.rand(17, 20), dtype=numpy.float32)
A = _cunumpy.array_to_cuda(a)
print "A=", A

# create another matrix
b = numpy.array(numpy.random.rand(20, 50), dtype=numpy.float32)
B = _cunumpy.array_to_cuda(b)
print "B=", B

# create a third target matrix
c = numpy.zeros((17,50),dtype=numpy.float32)
C = _cunumpy.array_to_cuda(c)
print "C=", C


# do sgemm - as plain dot product
D = _cublas.sgemm('n', 'n', 1.0, A, B, 0.0, C)
print D
print "D==C", D==C
    
# store result
r = _cunumpy.cuda_to_array(D)

# do gold standard dot product
s = numpy.dot(a,b)

print "e =", numpy.sum(numpy.abs(r-s))


# now present misshaped arrays
E = _cublas.sgemm('n', 'n', 1.0, B, A, 0.0, C)
print E
