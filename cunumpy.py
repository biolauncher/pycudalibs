# Copyright (c) 2009 Simon Beaumont <seb@modelsciences.com> - All Rights Reserved

import _cunumpy
import _cublas
import numpy


#
# extend _cunumpy.array with cublas methods
#

class CudaArray(_cunumpy.array):
    """
    Encapsulates device based arrays and some BLAS based linear algebra in numpy style.
    """
    def dot(self, other):
        """
        dot product of self with other using cublas.sgemm
        """
        n = self.shape[0]
        k = other.shape[1]
        c = zeros([n,k])
        # complex and double precison?
        return _cublas.sgemm('n','n',1.0, self, other, 0.0, c)


#
# functions to provide numpy like factory methods for cunumpy.array objects
#

def array(*args, **keyw):
    return CudaArray(numpy.array(*args, **keyw))

def arange(*args, **keyw):
    return CudaArray(numpy.arange(*args, **keyw))

def eye(*args, **keyw):
    return CudaArray(numpy.eye(*args, **keyw))

def zeros(*args, **keyw):
    return CudaArray(numpy.zeros(*args, **keyw))

def ones(*args, **keyw):
    return CudaArray(numpy.ones(*args, **keyw))



