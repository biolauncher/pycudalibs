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

array_types = [float32, float64, complex64, complex128]

#
# extend _cunumpy.array
#
class CudaArray(_cunumpy.array):
    """
    Encapsulates CUDA device based arrays and some CUBLAS based linear algebra in numpy style.
    """
    pass

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



