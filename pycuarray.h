/*
Copyright (C) 2009 Model Sciences Ltd.

This file is part of pycudalibs

    pycudalibs is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    pycudalibs is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the Lesser GNU General Public
    License along with pycudalibs.  If not, see <http://www.gnu.org/licenses/>.  
*/


/* 
 * Defines the cuda.array type and module init
 */

#if defined(_PYCUARRAY_H)
#else
#define _PYCUARRAY_H
 
#include <Python.h>

/* custom ML kernels */
#ifdef CUDAML
#include <cudaml.h>
#endif

/* cuda memory */
#include <pycumem.h>

/* numpy */
#include <structmember.h>
#include <numpy/arrayobject.h>

/* utils */
#include <pylibs.h>


#define DEVICE_ARRAY_MAXDIMS 2 
#define DEVICE_ARRAY_MINDIMS 1

#define DEVICE_ARRAY_TYPE_OK(dt) \
  ((PyTypeNum_ISCOMPLEX((dt)->type_num) && ((dt)->elsize == 8 || (dt)->elsize == 16)) \
   || (PyTypeNum_ISFLOAT((dt)->type_num) && ((dt)->elsize == 4 || (dt)->elsize == 8)))


/** 
 * A python base class to encapsulate a CUDA memory backed array
 * numpy is our inspiration and guiding light.
 * Fortran array semantics apply.
 * Vectors are conventionally column vectors.
 */

typedef struct {
  PyObject_HEAD
  cuda_Memory* d_mem;
  int a_ndims;                       /* number of dimensions */
  int a_dims[DEVICE_ARRAY_MAXDIMS];  /* dimensions */
  int e_size;                        /* sizeof element */
  PyArray_Descr* a_dtype;            /* keep reference to numpy dtype */
  int a_transposed;                  /* is array transposed */
} cuda_Array;


#define OP(A) ((A)->a_transposed ? 't' : 'n')

static inline int isvector(cuda_Array* d) {
  return d->a_ndims == 1 ? 1 : 0;
}

static inline int ismatrix(cuda_Array* d) {
  return d->a_ndims == 2 ? 1 : 0;
}

static inline int iscomplex(cuda_Array* d) {
  return PyTypeNum_ISCOMPLEX(d->a_dtype->type_num);
}

static inline int isdouble(cuda_Array* d) {
  return iscomplex(d) ? d->e_size == 16 : d->e_size == 8;
}

/* return number of elements: matrix, vector or scalar */

static inline int a_elements(cuda_Array* d) {
  return (d->a_ndims == 2) ? (d->a_dims[0] * d->a_dims[1]) : (d->a_ndims == 1 ? d->a_dims[0] : 1);
}

/* return size in bytes of array */
static inline size_t a_size(cuda_Array* a) {
  return (size_t) a_elements(a) * a->e_size;
}

/* static function prototypes */
static inline cuda_Array* make_vector(int, PyArray_Descr*);
static inline cuda_Array* make_matrix(int, int, PyArray_Descr*);
static inline cuda_Array* copy_array(cuda_Array*);
static inline PyArray_Descr* dtype(int);
static inline void* copy_devmem(cuda_Array*);
static inline void* copy_devmem2d(cuda_Array*);

#define UNARY_FUNCTION_PROTOTYPE(FUNC) static PyObject* cuda_Array_##FUNC(cuda_Array *);

#define UNARY_FUNCTION_METHOD_TABLE_ENTRY(FUNC) \
  {#FUNC, (PyCFunction) cuda_Array_##FUNC, METH_NOARGS, "elementwise " #FUNC " of an array"}

// declare all cuda_Array methods here 
static PyObject* cuda_Array_dot(cuda_Array*, PyObject*);
static PyObject* cuda_Array_transpose(cuda_Array*, PyObject*);
static PyObject* cuda_Array_scale(cuda_Array*, PyObject*);
static PyObject* cuda_Array_copy(cuda_Array*);
static PyObject* cuda_Array_2norm(cuda_Array*);
static PyObject* cuda_Array_asum(cuda_Array*);
static PyObject* cuda_Array_reshape(cuda_Array*, PyObject*);

#ifdef CULA // LAPACK
static PyObject* cuda_Array_svd(cuda_Array*, PyObject*, PyObject*);
static PyObject* cuda_Array_eigensystem(cuda_Array*, PyObject*, PyObject*);
static PyObject* cuda_Array_conjugateTranspose(cuda_Array*);
#endif // CULA

#ifdef CUDAML
static PyObject* cuda_Array_sum(cuda_Array*);
static PyObject* cuda_Array_max(cuda_Array*);
static PyObject* cuda_Array_min(cuda_Array*);
static PyObject* cuda_Array_product(cuda_Array*);

static PyObject* cuda_Array_csum(cuda_Array*);
static PyObject* cuda_Array_cmax(cuda_Array*);
static PyObject* cuda_Array_cmin(cuda_Array*);
static PyObject* cuda_Array_cproduct(cuda_Array*);

static PyObject* cuda_Array_esum(cuda_Array*, PyObject*);
static PyObject* cuda_Array_emul(cuda_Array*, PyObject*);
static PyObject* cuda_Array_epow(cuda_Array*, PyObject*);

UNARY_FUNCTION_PROTOTYPE(sqrt)
UNARY_FUNCTION_PROTOTYPE(log)
UNARY_FUNCTION_PROTOTYPE(log2)
UNARY_FUNCTION_PROTOTYPE(log10)

UNARY_FUNCTION_PROTOTYPE(sin)
UNARY_FUNCTION_PROTOTYPE(cos)
UNARY_FUNCTION_PROTOTYPE(tan)

UNARY_FUNCTION_PROTOTYPE(sinh)
UNARY_FUNCTION_PROTOTYPE(cosh)
UNARY_FUNCTION_PROTOTYPE(tanh)

UNARY_FUNCTION_PROTOTYPE(exp)
UNARY_FUNCTION_PROTOTYPE(exp10)

UNARY_FUNCTION_PROTOTYPE(sinpi)
UNARY_FUNCTION_PROTOTYPE(cospi)

UNARY_FUNCTION_PROTOTYPE(asin)
UNARY_FUNCTION_PROTOTYPE(acos)
UNARY_FUNCTION_PROTOTYPE(atan)
UNARY_FUNCTION_PROTOTYPE(asinh)
UNARY_FUNCTION_PROTOTYPE(acosh)
UNARY_FUNCTION_PROTOTYPE(atanh)

UNARY_FUNCTION_PROTOTYPE(erf)
UNARY_FUNCTION_PROTOTYPE(erfc)
UNARY_FUNCTION_PROTOTYPE(erfinv)
UNARY_FUNCTION_PROTOTYPE(erfcinv)
UNARY_FUNCTION_PROTOTYPE(lgamma)
UNARY_FUNCTION_PROTOTYPE(tgamma)

UNARY_FUNCTION_PROTOTYPE(trunc)
UNARY_FUNCTION_PROTOTYPE(round)
UNARY_FUNCTION_PROTOTYPE(rint)
UNARY_FUNCTION_PROTOTYPE(floor)
UNARY_FUNCTION_PROTOTYPE(ceil)

#endif // CUDAML

#endif
