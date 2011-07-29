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
 * Defines the _cunumpy module which provides a cuda array type along the lines of numpy
 */

#if defined(_PYCUNUMPY_H)
#else
#define _PYCUNUMPY_H 1

#include <Python.h>

/* module exception object */
extern PyObject* cuda_exception;

/* moudle types */
extern PyTypeObject* cuda_ArrayType;
 
/* cuda array object */
//#include <pycuarray.h>

/* module importer */

#define CUDA_MODULE_NAME "_cunumpy"

/* type constants */
#define CUDA_ARRAY_TYPE_NAME "cuda.array"
#define CUDA_ARRAY_TYPE_SYM_NAME "array"

#define CUDA_ERROR_TYPE_NAME "cuda.CUDAERROR"
#define CUDA_ERROR_TYPE_SYM_NAME "CUDAERROR"

#define CUDA_MEMORY_TYPE_NAME "cuda.memory"

static inline int import_cunumpy(void) {
  PyObject* module = PyImport_ImportModule(CUDA_MODULE_NAME);
  
  if (module != NULL) {
    cuda_ArrayType = (PyTypeObject *)PyObject_GetAttrString(module, CUDA_ARRAY_TYPE_SYM_NAME);
    if (cuda_ArrayType == NULL) return -1;

    cuda_exception = PyObject_GetAttrString(module, CUDA_ERROR_TYPE_SYM_NAME);
    if (cuda_exception == NULL) return -1;
  } 
  return 0;
}

#endif


