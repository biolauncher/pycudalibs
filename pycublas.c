/* Copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/**
 * Python integration to CUDA BLAS routines. This low level API module
 * is called _cublas and has an associated Python wrapper module.
 */

#include <pycublas.h>
#include <pycunumpy.h>


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

//static PyObject* cublas_exception; // why are we not using this?

/* initialise the Python c extension module - this function has to be named consistently */

PyMODINIT_FUNC init_cublas(void) {
  // initialise the module
  PyObject* module = Py_InitModule("_cublas", _cublas_methods);
  if (module == NULL) return;
  import_cunumpy();
}


/** 
 * Low level CUDA BLAS library api.
 */

static PyObject* init(PyObject* self, PyObject* args) {
  if (!PyArg_ParseTuple(args, "")) 
    return NULL;
  else if (cuda_error(cublasInit(), "cublasInit"))
    return NULL;
  else 
    return Py_BuildValue("");
}

static PyObject* shutdown(PyObject* self, PyObject* args) {
  if (!PyArg_ParseTuple(args, "")) 
    return NULL;
  else if (cuda_error(cublasShutdown(), "cublasShutdown"))
    return NULL;
  else
    return Py_BuildValue("");
}


/*
We take the input matrices A, B and C as well as the scalars and
transposition keys - We know the dimensions of the input objects as
these are required to be DeviceMemory arrays.

char transa, char transb, float alpha, DeviceMemory A, deviceMemory B,
float beta, deviceMemory C 
return the updated array C 
*/

static PyObject* sgemm(PyObject* self, PyObject* args) {
  cuda_DeviceMemory *A, *B, *C;
  float alpha, beta;
  char transa, transb;

  if (PyArg_ParseTuple(args, "ccfO!O!fO!", 
                       &transa, &transb, &alpha, 
                       cuda_DeviceMemoryType, &A, 
                       cuda_DeviceMemoryType, &B, 
                       &beta, cuda_DeviceMemoryType, &C)) {

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];
    int ldc = C->a_dims[0];

    // need todo some shape checking here! 
    if (A->a_dims[1] != B->a_dims[0] ||
        A->a_dims[0] != C->a_dims[0] ||
        C->a_dims[1] != B->a_dims[1]) {

      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }
    // and trans? dimension swapping?
    // what happens to ld when transposing? normally in fortran matrix rep it is row size i.e. dims[0]
    // nothing me thinks we present matrices as is
    // then shouldn't this be the case for other dimensions?

    /*
      void 
      cublasSgemm (char transa, char transb, int m, int n, 
                   int k, float alpha, const float *A, int lda, 
                   const float *B, int ldb, float beta, 
                   float *C, int ldc)
    */
    cublasSgemm(transa, transb, A->a_dims[0], B->a_dims[1], A->a_dims[1], 
                alpha, A->d_ptr, lda, B->d_ptr, ldb, beta, C->d_ptr, ldc);

    if (cublas_error("sgemm")) 
      return NULL;
    else 
      return Py_BuildValue("O", C);
  
  } else {
    return NULL;
  }
}


