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
        # 1a check array is of permitted types
        if not self.dtype in array_types:
            raise ValueError("array does not have supported dtype")

        # 1b check type of other is same as self - may relax this one day
        if not isinstance(other, _cunumpy.array) or not self.dtype == other.dtype:
            raise ValueError("argument array is not a compatible dtype")
            
        # 2. vector-*
        if self.ndim == 1:

            if other.ndim == 1:
                # vector-vector
                if self.dtype.kind == 'f':

                    if self.itemsize == 4:
                        return _cublas.sdot(self, other)
                    elif self.itemsize == 8:
                        return _cublas.ddot(self, other)

                elif self.dtype.kind == 'c':
                    if self.itemsize == 8:
                        return _cublas.cdotu(self, other)
                    if self.itemsize == 16:
                        raise NotImplementedError("BLAS1 complex double vector dot product not implemented!")

            elif other.ndim == 2:
                # vector-matrix
                if self.dtype.kind == 'f':

                    if self.itemsize == 4:
                        raise NotImplementedError("real single vector-matrix mutiply not implemented!")
                    elif self.itemsize == 8:
                        raise NotImplementedError("real double vector-matrix mutiply not implemented!")

                elif self.dtype.kind == 'c':
                    if self.itemsize == 8:
                        raise NotImplementedError("complex single vector-matrix mutiply not implemented!")
                    if self.itemsize == 16:
                        raise NotImplementedError("complex double vector-matrix mutiply not implemented!")
                
            else:
                raise ValueError("argument array has invalid number of dimensions for vector dot product")

        # matrix-*
        elif self.ndim == 2:

            if other.ndim == 1:

                n = self.shape[0]
                k = 1
                c = zeros([n], dtype=self.dtype)

                # matrix-vector
                if self.dtype.kind == 'f':

                    if self.itemsize == 4:
                        return _cublas.sgemv('n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 8:
                        return _cublas.dgemv('n', 1.0, self, other, 0.0, c)

                elif self.dtype.kind == 'c':
                    if self.itemsize == 8:
                        raise NotImplementedError("BLAS2 complex single matrix-vector mutiply not implemented!")
                    if self.itemsize == 16:
                        raise NotImplementedError("BLAS2 complex double matrix-vector mutiply not implemented!")


            elif other.ndim == 2:
                # 3. matrix-matrix: construct a suitable result array
                n = self.shape[0]
                k = other.shape[1]
                c = zeros([n,k], dtype=self.dtype)

                # 4. select appropriate blas routine based on shape/kind
                if self.dtype.kind == 'c':

                    if self.itemsize == 8:
                        return _cublas.cgemm('n', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 16:
                        return _cublas.zgemm('n', 'n', 1.0, self, other, 0.0, c)

                elif self.dtype.kind == 'f':

                    if self.itemsize == 4:
                        return _cublas.sgemm('n', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 8:
                        return _cublas.dgemm('n', 'n', 1.0, self, other, 0.0, c)

            else:
                raise ValueError("argument array has invalid number of dimensions for matrix dot product")

        else:
            raise ValueError("array has invalid number of dimensions for matrix dot product")

    def dot2(self, other):
        """
        inner product of self with other using cublas.sgemm treats everything as arrays
        """
        # 1a check array is of permitted types
        if not self.dtype in array_types:
            raise ValueError("array does not have supported dtype")

        # 1b check type of other is same as self - may relax this one day
        if not isinstance(other, _cunumpy.array) or not self.dtype == other.dtype:
            raise ValueError("argument array is not a compatible dtype")
        
        # 2. vector-*
        # we transpose default (conventional) column vectors into left row vector for the operation

        if self.ndim == 1:
            n = 1

            if other.ndim == 1:
                # vector-vector - need to return a scalar

                k = 1
                c = zeros([n,k], dtype=self.dtype)

                if self.dtype.kind == 'f':
                    if self.itemsize == 4:
                        return _cublas.sgemm('t', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 8:
                        return _cublas.dgemm('t', 'n', 1.0, self, other, 0.0, c)


                elif self.dtype.kind == 'c':
                    if self.itemsize == 8:
                        return _cublas.cgemm('t', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 16:
                        return _cublas.zgemm('t', 'n', 1.0, self, other, 0.0, c)

            elif other.ndim == 2:
                # vector-matrix - need to construct a standard vector
                
                k = other.shape[1]
                c = zeros([n,k], dtype=self.dtype)

                if self.dtype.kind == 'f':

                    if self.itemsize == 4:
                        return _cublas.sgemm('t', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 8:
                        return _cublas.dgemm('t', 'n', 1.0, self, other, 0.0, c)


                elif self.dtype.kind == 'c':
                    if self.itemsize == 8:
                        return _cublas.cgemm('t', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 16:
                        return _cublas.zgemm('t', 'n', 1.0, self, other, 0.0, c)
                
            else:
                raise ValueError("argument array has invalid number of dimensions for vector dot product")

        # matrix-*
        elif self.ndim == 2:
            n = self.shape[0]
            
            if other.ndim == 1:

                # matrix-vector - we arrange to return a row vector here as this is 
                # the numpy (and our) convention. See vector-matrix tricks above.
                k = 1
                c = zeros([n], dtype=self.dtype)

                if self.dtype.kind == 'f':

                    if self.itemsize == 4:
                        return _cublas.sgemm('n', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 8:
                        return _cublas.dgemm('n', 'n', 1.0, self, other, 0.0, c)


                elif self.dtype.kind == 'c':
                    if self.itemsize == 8:
                        return _cublas.cgemm('n', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 16:
                        return _cublas.zgemm('n', 'n', 1.0, self, other, 0.0, c)


            elif other.ndim == 2:
                # matrix-matrix

                k = other.shape[1]
                c = zeros([n,k], dtype=self.dtype)

                # 4. select appropriate blas routine based on shape/kind
                if self.dtype.kind == 'c':

                    if self.itemsize == 8:
                        return _cublas.cgemm('n', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 16:
                        return _cublas.zgemm('n', 'n', 1.0, self, other, 0.0, c)

                elif self.dtype.kind == 'f':

                    if self.itemsize == 4:
                        return _cublas.sgemm('n', 'n', 1.0, self, other, 0.0, c)
                    elif self.itemsize == 8:
                        return _cublas.dgemm('n', 'n', 1.0, self, other, 0.0, c)

            else:
                raise ValueError("argument array has invalid number of dimensions for matrix dot product")

        else:
            raise ValueError("array has invalid number of dimensions for matrix dot product")

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



