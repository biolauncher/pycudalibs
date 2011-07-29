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
 * Defines the cuda.array type.
 */

#if defined(_PYCUARRAY_H)
#else
#define _PYCUARRAY_H
 
#include <Python.h>

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

/* static function prototypes */
static inline cuda_Array* make_vector(int, PyArray_Descr*);
static inline cuda_Array* make_matrix(int, int, PyArray_Descr*);
static inline cuda_Array* copy_array(cuda_Array*);

// TODO declare all methods in here - 
static PyObject* cuda_Array_dot(cuda_Array*, PyObject*);
static PyObject* cuda_Array_transpose(cuda_Array*, PyObject*);
static PyObject* cuda_Array_scale(cuda_Array*, PyObject*);
static PyObject* cuda_Array_copy(cuda_Array*);
static PyObject* cuda_Array_2norm(cuda_Array*);
static PyObject* cuda_Array_asum(cuda_Array*);
static PyObject* cuda_Array_reshape(cuda_Array*, PyObject*);


#endif
