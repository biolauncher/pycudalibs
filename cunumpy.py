# Copyright (c) 2009 Simon Beaumont <seb@modelsciences.com> - All Rights Reserved

import _cunumpy
import _cublas
import numpy

#
# the supported dtype(s)
#
float32 = numpy.float32
float64 = numpy.float64
complex64 = numpy.complex64
complex128 = numpy.complex128


#
# extend _cunumpy.array with cublas methods
#
class CudaArray(_cunumpy.array):
    """
    Encapsulates CUDA device based arrays and some CUBLAS based linear algebra in numpy style.
    """
    def dot(self, other):
        """
        inner product of self with other using cublas.sgemm and friends
        """
        # 1. check type of other is same as self - may relax this one day
        if not isinstance(other, _cunumpy.array) or not self.dtype == other.dtype:
            raise TypeError("argument array is not a compatible type")
            
        # 2. make sure shapes are compatible - or just let the lower levels barf?
        #m = self.shape[1]
        #if m != other.shape[0]: 
        #    raise TypeError... 
        # 3. construct a suitable result array
        n = self.shape[0]
        k = other.shape[1]
        c = zeros([n,k], dtype=self.dtype)
        # 4. select appropriate blas routine based on shape/kind
        return _cublas.sgemm('n','n',1.0, self, other, 0.0, c)


#
# functions to provide numpy like factory methods for cunumpy.array objects
#

def array(*args, **keyw):
    return CudaArray(numpy.array(*args, **keyw), **keyw)

def arange(*args, **keyw):
    return CudaArray(numpy.arange(*args, **keyw), **keyw)

def eye(*args, **keyw):
    return CudaArray(numpy.eye(*args, **keyw), **keyw)

def zeros(*args, **keyw):
    return CudaArray(numpy.zeros(*args, **keyw), **keyw)

def ones(*args, **keyw):
    return CudaArray(numpy.ones(*args, **keyw), **keyw)

def identity(*args, **keyw):
    return CudaArray(numpy.identity(*args, **keyw), **keyw)



