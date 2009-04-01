/* Copyright (c) 2009 Simon Beaumont - All Rights Reserved */

#if defined(_PYCUBLAS_H)
#else
#define _PYCUBLAS_H 1
#include <pycuda.h>

/* Python callable functions in the CUDA BLAS api */

static PyObject* init(PyObject* self, PyObject* args); 
static PyObject* shutdown(PyObject* self, PyObject* args); 
static PyObject* sgemm(PyObject* self, PyObject* args);


#endif
