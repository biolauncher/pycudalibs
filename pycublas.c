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
   "Releases CPU‐side resources used by the CUBLAS library."},
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


/*
Function cublasSgemm()
----------------------

void 
cublasSgemm (char transa, char transb, int m, int n, 
             int k, float alpha, const float *A, int lda, 
             const float *B, int ldb, float beta, 
             float *C, int ldc)

computes the product of matrix A and matrix B, multiplies the result 
by scalar alpha, and adds the sum to the product of matrix C and
scalar beta. It performs one of the matrix-matrix operations:

    C = alpha * op(A) * op(B) + beta * C,  
    where op(X) = X or op(X) = transpose(X),

and alpha and beta are single-precision scalars. A, B and C are 
matrices consisting of single-precision elements, with op(A) an m x k 
matrix, op(B) a k x n matrix, and C an m x n matrix. Matrices A, B, and C
are stored in column-major format, and lda, ldb, and ldc are the 
leading dimensions of the two-dimensional arrays containing A, B, and
C.

Input
-----
transa specifies op(A). If transa == 'N' or 'n', op(A) = A. 
       If transa == 'T', 't', 'C', or 'c', op(A) = transpose(A).
transb specifies op(B). If transb == 'N' or 'n', op(B) = B. 
       If transb == 'T', 't', 'C', or 'c', op(B) = transpose(B).
m      number of rows of matrix op(A) and rows of matrix C; m must be at
       least zero.
n      number of columns of matrix op(B) and number of columns of C; 
       n must be at least zero.
k      number of columns of matrix op(A) and number of rows of op(B);
       k must be at least zero.
alpha  single-precision scalar multiplier applied to op(A) * op(B).
A      single-precision array of dimensions (lda, k) if transa == 'N' or 
       'n', and of dimensions (lda, m) otherwise. If transa == 'N' or 
       'n' lda must be at least max(1, m), otherwise lda must be at least
       max(1, k).
lda    leading dimension of two-dimensional array used to store matrix A.
B      single-precision array of dimensions (ldb, n) if transb == 'N' or
       'n', and of dimensions (ldb, k) otherwise. If transb == 'N' or 
       'n' ldb must be at least max (1, k), otherwise ldb must be at least
       max(1, n).
ldb    leading dimension of two-dimensional array used to store matrix B.
beta   single-precision scalar multiplier applied to C. If zero, C does not 
       have to be a valid input
C      single-precision array of dimensions (ldc, n); ldc must be at least
       max(1, m).
ldc    leading dimension of two-dimensional array used to store matrix C.

Output
------
C      updated based on C = alpha * op(A)*op(B) + beta * C.

Reference: http://www.netlib.org/blas/sgemm.f

Error status for this function can be retrieved via cublasGetError().

Error Status
------------
CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS was not initialized
CUBLAS_STATUS_INVALID_VALUE    if m < 0, n < 0, or k < 0
CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU


 */

static PyObject* sgemm(PyObject* self, PyObject* args) {
  // get the matrix and alpha and beta (optional args) copy em all up to the device
  return Py_BuildValue("i", 42);
}
