/*
Copyright (C) 2009-2011 Model Sciences Ltd.

This file is part of pycudalibs

    pycudalibs is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    Pycudalibs is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the Lesser GNU General Public
    License along with pycudalibs.  If not, see <http://www.gnu.org/licenses/>.  
*/

/**
 * numpy integration with cuda device memory, defines module: _cunumpy
 */


/*******************
 * module functions
 ******************/

#include <pycunumpy.h>

static PyMethodDef module_methods[] = {
  {NULL}  /* Sentinel */
};

/* module exception object */
PyObject* cuda_exception;

/* module types */
PyTypeObject* cuda_ArrayType;
 

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
init_cunumpy(void) {
    PyObject* module;

    if (PyType_Ready(&cuda_ArrayType) < 0)
      return;

    if (init_cuda_MemoryType() < 0)
      return;

    module = Py_InitModule3(CUDA_MODULE_NAME, module_methods, "CUDA numpy style array module.");

    if (module == NULL) return;
    
    else {
      Py_INCREF(&cuda_ArrayType);
      PyModule_AddObject(module, CUDA_ARRAY_TYPE_SYM_NAME, (PyObject *)&cuda_ArrayType);

      cuda_exception = PyErr_NewException(CUDA_ERROR_TYPE_NAME, NULL, NULL);
      Py_INCREF(cuda_exception);
      PyModule_AddObject(module, CUDA_ERROR_TYPE_SYM_NAME, cuda_exception);

      import_array(); // import numpy module
    }
}
