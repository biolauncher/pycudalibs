/* Copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/**
 * Python integration to CUDA BLAS routines. This low level API module
 * is called _cublas and has an associated Python wrapper module.
 */

#include <pycublas.h>
#include <stdio.h>

static PyMethodDef _cublas_methods[] = {
  {"init", init, METH_VARARGS, 
   "Initialise CUBLAS library by attaching to CUDA device that is bound to the calling host thread."},
  {"sgemm", sgemm, METH_VARARGS,
   "Single Pecision BLAS3: C = alpha * op(A) * op(B) + beta * C"},
  {"close", shutdown, METH_VARARGS,
   "Releases CPU‐side resources used by the CUBLAS library. The release of GPU‐side resources may be deferred until the application shuts down."}, 
  {NULL, NULL, 0, NULL}
};

/* define an exception object for cublas */

static PyObject* cublas_exception;

/* initialise the Python c extension module - this function has to be named consistently */

PyMODINIT_FUNC init_cublas(void) {
  // initialise the module
  PyObject* module = Py_InitModule("_cublas", _cublas_methods);
  if (module == NULL) return;
  
  else {
    cublas_exception = PyErr_NewException("cublas.error", NULL, NULL);
    Py_INCREF(cublas_exception);
    PyModule_AddObject(module, "error", cublas_exception);
  } 
}

/* cublas library error to Python exception */

static inline int cublas_error(cublasStatus status, char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    PyErr_SetString(cublas_exception, get_cublas_error_text(status));
#if DEBUG > 0
    fprintf(stderr, "bad vibes from CUDA: %s=%d\n", what, status);
#endif
    return 1;
    
  } else {
#if DEBUG > 0
    fprintf(stderr, "%s: looking good in the CUDA device!\n", what);
#endif
    return 0;
  }
}


/** 
 * Low level CUDA BLAS library api.
 */

static PyObject* init(PyObject* self, PyObject* args) {
  if (!PyArg_ParseTuple(args, "")) 
    return NULL;
  else if (cublas_error(cublasInit(), "cublasInit"))
    return NULL;
  else 
    return Py_BuildValue("");
}

static PyObject* shutdown(PyObject* self, PyObject* args) {
  if (!PyArg_ParseTuple(args, "")) 
    return NULL;
  else if (cublas_error(cublasShutdown(), "cublasShutdown"))
    return NULL;
  else
    return Py_BuildValue("");
}


static PyObject* sgemm(PyObject* self, PyObject* args) {
  // get the matrix and alpha and beta (optional args) copy em all up to the device
  return Py_BuildValue("i", 42);
}
