/* Copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/**
 * Python integration to CUDA BLAS routines.
 *  defines module: _cublas 
 */
#define NO_IMPORT_ARRAY
#include <pycublas.h>


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
sgemm - single precision real matrix-matrix multiply
transa, transb, alpha, A, B, beta, C
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

    // do some shape checking here
    if (A->a_dims[1] != B->a_dims[0] ||
        A->a_dims[0] != C->a_dims[0] ||
        C->a_dims[1] != B->a_dims[1]) {

      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }
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


/*
dgemm - double precision real matrix-matrix multiply
transa, transb, alpha, A, B, beta, C
return the updated array C 
*/

static PyObject* dgemm(PyObject* self, PyObject* args) {
  cuda_DeviceMemory *A, *B, *C;
  double alpha, beta;
  char transa, transb;

  if (PyArg_ParseTuple(args, "ccdO!O!dO!", 
                       &transa, &transb, &alpha, 
                       cuda_DeviceMemoryType, &A, 
                       cuda_DeviceMemoryType, &B, 
                       &beta, cuda_DeviceMemoryType, &C)) {

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];
    int ldc = C->a_dims[0];

    // do some shape checking here
    if (A->a_dims[1] != B->a_dims[0] ||
        A->a_dims[0] != C->a_dims[0] ||
        C->a_dims[1] != B->a_dims[1]) {

      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }
    /*
      void 
      cublasDgemm (char transa, char transb, int m, int n, 
                   int k, float double, const double *A, int lda, 
                   const double *B, int ldb, double beta, 
                   double *C, int ldc)
    */
    cublasDgemm(transa, transb, A->a_dims[0], B->a_dims[1], A->a_dims[1], 
                alpha, A->d_ptr, lda, B->d_ptr, ldb, beta, C->d_ptr, ldc);

    if (cublas_error("dgemm")) 
      return NULL;
    else 
      return Py_BuildValue("O", C);
  
  } else {
    return NULL;
  }
}


/*
cgemm - single precision complex matrix-matrix multiply
transa, transb, alpha, A, B, beta, C
return the updated array C 
*/

static PyObject* cgemm(PyObject* self, PyObject* args) {
  cuda_DeviceMemory *A, *B, *C;
  Py_complex alpha, beta;
  char transa, transb;

  if (PyArg_ParseTuple(args, "ccDO!O!DO!", 
                       &transa, &transb, &alpha, 
                       cuda_DeviceMemoryType, &A, 
                       cuda_DeviceMemoryType, &B, 
                       &beta, cuda_DeviceMemoryType, &C)) {

    /* N.B. unfortunately python only has a double precision complex type so we must cast with risk
       here - in most of our usage of this routine we use 0 and 1 as parameters but this comment
       is the only warning */

    cuComplex c_alpha;
    cuComplex c_beta;

    c_alpha.x = (float) alpha.real;
    c_alpha.y = (float) alpha.imag;
    c_beta.x = (float) alpha.real;
    c_beta.y = (float) alpha.imag;

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];
    int ldc = C->a_dims[0];

    // do some shape checking here
    if (A->a_dims[1] != B->a_dims[0] ||
        A->a_dims[0] != C->a_dims[0] ||
        C->a_dims[1] != B->a_dims[1]) {

      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }
    /*
      void 
      cublasCgemm (char transa, char transb, int m, int n, 
                   int k, cuComplex alpha, const cuComplex *A, 
                   int lda, const cuComplex *B, int ldb, 
                   cuComplex beta, cuComplex *C, int ldc) 
    */
    cublasCgemm(transa, transb, A->a_dims[0], B->a_dims[1], A->a_dims[1], 
                c_alpha, A->d_ptr, lda, B->d_ptr, ldb, c_beta, C->d_ptr, ldc);

    if (cublas_error("cgemm")) 
      return NULL;
    else 
      return Py_BuildValue("O", C);
  
  } else {
    return NULL;
  }
}


/*
zgemm - double precision complex matrix-matrix multiply
transa, transb, alpha, A, B, beta, C
return the updated array C 
*/

static PyObject* zgemm(PyObject* self, PyObject* args) {
  cuda_DeviceMemory *A, *B, *C;
  Py_complex alpha, beta;
  char transa, transb;

  if (PyArg_ParseTuple(args, "ccDO!O!DO!", 
                       &transa, &transb, &alpha, 
                       cuda_DeviceMemoryType, &A, 
                       cuda_DeviceMemoryType, &B, 
                       &beta, cuda_DeviceMemoryType, &C)) {

    cuDoubleComplex c_alpha;
    cuDoubleComplex c_beta;

    c_alpha.x = alpha.real;
    c_alpha.y = alpha.imag;
    c_beta.x = alpha.real;
    c_beta.y = alpha.imag;

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];
    int ldc = C->a_dims[0];

    // do some shape checking here
    if (A->a_dims[1] != B->a_dims[0] ||
        A->a_dims[0] != C->a_dims[0] ||
        C->a_dims[1] != B->a_dims[1]) {

      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }
    /*
      void 
      cublasZgemm (char transa, char transb, int m, int n, 
                   int k, cuComplex alpha, const cuDoubleComplex *A, 
                   int lda, const cuDoubleComplex *B, int ldb, 
                   cuDoubleComplex beta, cuDoubleComplex *C, int ldc) 
    */
    cublasZgemm(transa, transb, A->a_dims[0], B->a_dims[1], A->a_dims[1], 
                c_alpha, A->d_ptr, lda, B->d_ptr, ldb, c_beta, C->d_ptr, ldc);

    if (cublas_error("zgemm")) 
      return NULL;
    else 
      return Py_BuildValue("O", C);
  
  } else {
    return NULL;
  }
}

/*
sdot - single precision real vector-vector dot product
returns a python float
*/

static PyObject* sdot(PyObject* self, PyObject* args) {
  cuda_DeviceMemory *A, *B;
    
  if (PyArg_ParseTuple(args, "O!O!", 
                       cuda_DeviceMemoryType, &A, 
                       cuda_DeviceMemoryType, &B)) {

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];

    // do some shape checking here
    if (lda != ldb) {
      PyErr_SetString(PyExc_ValueError, "vectors have different dimensions");
      return NULL;
    }
    /*
      float 
      cublasSdot (int n, const float *x, int incx, const float *y, int incy) 
    */

    float ip = cublasSdot(lda, A->d_ptr, 1, B->d_ptr, 1);

    if (cublas_error("sdot")) 
      return NULL;
    else 
      return Py_BuildValue("f", ip);
  
  } else {
    return NULL;
  }
}

/*
ddot - double precision real vector-vector dot product
returns a python double
*/

static PyObject* ddot(PyObject* self, PyObject* args) {
  cuda_DeviceMemory *A, *B;
    
  if (PyArg_ParseTuple(args, "O!O!", 
                       cuda_DeviceMemoryType, &A, 
                       cuda_DeviceMemoryType, &B)) {

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];

    // do some shape checking here
    if (lda != ldb) {
      PyErr_SetString(PyExc_ValueError, "vectors have different dimensions");
      return NULL;
    }

    /*
      double
      cublasDdot (int n, const double *x, int incx, const double *y, int incy) 
    */

    double ip = cublasDdot(lda, A->d_ptr, 1, B->d_ptr, 1);

    if (cublas_error("ddot")) 
      return NULL;
    else 
      return Py_BuildValue("d", ip);
  
  } else {
    return NULL;
  }
}


/*
cdotu - single precision complex vector-vector dot product
returns a python complex
*/

static PyObject* cdotu(PyObject* self, PyObject* args) {
  cuda_DeviceMemory *A, *B;
    
  if (PyArg_ParseTuple(args, "O!O!", 
                       cuda_DeviceMemoryType, &A, 
                       cuda_DeviceMemoryType, &B)) {

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];

    // do some shape checking here
    if (lda != ldb) {
      PyErr_SetString(PyExc_ValueError, "vectors have different dimensions");
      return NULL;
    }

    /*
      cuComplex 
      cublasCdotu (int n, const cuComplex *x, int incx, const cuComplex *y, int incy) 
    */

    cuComplex cip = cublasCdotu(lda, A->d_ptr, 1, B->d_ptr, 1);

    if (cublas_error("cdotu")) 
      return NULL;
    else {
      Py_complex ip;
      ip.real = cip.x;
      ip.imag = cip.y;
      return Py_BuildValue("D", ip);
    }
  } else {
    return NULL;
  }
}

/*
cdotc - single precision complex vector-vector dot product
takes conjugate of first vector (A)
returns a python complex
*/

static PyObject* cdotc(PyObject* self, PyObject* args) {
  cuda_DeviceMemory *A, *B;
    
  if (PyArg_ParseTuple(args, "O!O!", 
                       cuda_DeviceMemoryType, &A, 
                       cuda_DeviceMemoryType, &B)) {

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];

    // do some shape checking here
    if (lda != ldb) {
      PyErr_SetString(PyExc_ValueError, "vectors have different dimensions");
      return NULL;
    }

    /*
      cuComplex 
      cublasCdotc (int n, const cuComplex *x, int incx, const cuComplex *y, int incy) 
    */

    cuComplex cip = cublasCdotc(lda, A->d_ptr, 1, B->d_ptr, 1);

    if (cublas_error("cdotc")) 
      return NULL;
    else {
      Py_complex ip;
      ip.real = cip.x;
      ip.imag = cip.y;
      return Py_BuildValue("D", ip);
    }
  } else {
    return NULL;
  }
}


/************************
 * module function table
 ************************/

static PyMethodDef _cublas_methods[] = {
  {"init", init, METH_VARARGS, 
   "Initialise CUBLAS library by attaching to CUDA device that is bound to the calling host thread."},

  {"sgemm", sgemm, METH_VARARGS,
   "Single Precision BLAS3 float matrix multiply: C = alpha * op(A) * op(B) + beta * C"},
  {"cgemm", cgemm, METH_VARARGS,
   "Single Precision BLAS3 complex matrix multiply: C = alpha * op(A) * op(B) + beta * C"},
  {"dgemm", dgemm, METH_VARARGS,
   "Double Precision BLAS3 real matrix multiply: C = alpha * op(A) * op(B) + beta * C"},
  {"zgemm", zgemm, METH_VARARGS,
   "Double Precision BLAS3 complex matrix multiply: C = alpha * op(A) * op(B) + beta * C"},

  {"sdot", sdot, METH_VARARGS,
   "Single Precision BLAS1 real vector dot product"},
  {"ddot", ddot, METH_VARARGS,
   "Double Precision BLAS1 real vector dot product"},

  {"cdotu", cdotu, METH_VARARGS,
   "Double Precision BLAS1 complex vector dot product"},
  {"cdotc", cdotc, METH_VARARGS,
   "Double Precision BLAS1 complex vector transpose dot product "},


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
