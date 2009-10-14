/*
Copyright (C) 2009 Model Sciences Ltd.

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
 * Python integration to CUDA BLAS routines.
 *  defines module: _cublas 
 */
#define NO_IMPORT_ARRAY
#include <pycublas.h>

/** 
 * Low level CUDA BLAS library api. - there are bugs in the CUDA blas library that seem
 * to ignore iniialization - but best to use it where indicated. 
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
  cuda_Array *A, *B, *C;
  float alpha, beta;
  char transa, transb;

  if (PyArg_ParseTuple(args, "ccfO!O!fO!", 
                       &transa, &transb, &alpha, 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B, 
                       &beta, cuda_ArrayType, &C)) {

    // XXX new setup for index twiddling transpose - this now goes - but see below
    int m = transa == 't' ? A->a_dims[1] : A->a_dims[0];
    int k = transa == 't' ? A->a_dims[0] : A->a_dims[1]; 
    int kb = transb == 't' ? B->a_dims[1] : B->a_dims[0];
    int n = transb == 't' ? B->a_dims[0] : B->a_dims[1];
    int mc = C->a_dims[0];
    int nc = C->a_dims[1];

    // check geometry is good
    if (k != kb || m != mc || n != nc) {
      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }

    // XXX since transposed arrays will have had their dims swapped we need to swap again here
    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];
    int ldc = C->a_dims[0];

    /*
      void 
      cublasSgemm (char transa, char transb, int m, int n, 
                   int k, float alpha, const float *A, int lda, 
                   const float *B, int ldb, float beta, 
                   float *C, int ldc)
    */
    cublasSgemm(transa, transb, m, n, k, alpha, 
                A->d_mem->d_ptr, lda, B->d_mem->d_ptr, ldb, beta, C->d_mem->d_ptr, ldc);

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
  cuda_Array *A, *B, *C;
  double alpha, beta;
  char transa, transb;

  if (PyArg_ParseTuple(args, "ccdO!O!dO!", 
                       &transa, &transb, &alpha, 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B, 
                       &beta, cuda_ArrayType, &C)) {

    // new setup for index twiddling transpose
    int m = transa == 't' ? A->a_dims[1] : A->a_dims[0];
    int k = transa == 't' ? A->a_dims[0] : A->a_dims[1]; 
    int kb = transb == 't' ? B->a_dims[1] : B->a_dims[0];
    int n = transb == 't' ? B->a_dims[0] : B->a_dims[1];
    int mc = C->a_dims[0];
    int nc = C->a_dims[1];

    // check geometry is good
    if (k != kb || m != mc || n != nc) {
      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];
    int ldc = C->a_dims[0];

    
    /*
      void 
      cublasDgemm (char transa, char transb, int m, int n, 
                   int k, double alpha, const double *A, int lda, 
                   const double *B, int ldb, double beta, 
                   double *C, int ldc)
    */
    cublasDgemm(transa, transb, m, n, k, alpha, A->d_mem->d_ptr, lda, B->d_mem->d_ptr, ldb, beta, C->d_mem->d_ptr, ldc);

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
  cuda_Array *A, *B, *C;
  Py_complex alpha, beta;
  char transa, transb;

  if (PyArg_ParseTuple(args, "ccDO!O!DO!", 
                       &transa, &transb, &alpha, 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B, 
                       &beta, cuda_ArrayType, &C)) {

    /* N.B. unfortunately python only has a double precision complex
       type so we must cast with risk here - in most of our usage of
       this routine we use 0 and 1 as alpha and beta parameters but
       this comment is the only warning */

    cuComplex c_alpha;
    cuComplex c_beta;

    c_alpha.x = (float) alpha.real;
    c_alpha.y = (float) alpha.imag;
    c_beta.x = (float) alpha.real;
    c_beta.y = (float) alpha.imag;

    // new setup for index twiddling transpose
    int m = transa == 't' ? A->a_dims[1] : A->a_dims[0];
    int k = transa == 't' ? A->a_dims[0] : A->a_dims[1]; 
    int kb = transb == 't' ? B->a_dims[1] : B->a_dims[0];
    int n = transb == 't' ? B->a_dims[0] : B->a_dims[1];
    int mc = C->a_dims[0];
    int nc = C->a_dims[1];

    // check geometry is good
    if (k != kb || m != mc || n != nc) {
      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];
    int ldc = C->a_dims[0];

    /*
      void 
      cublasCgemm (char transa, char transb, int m, int n, 
                   int k, cuComplex alpha, const cuComplex *A, 
                   int lda, const cuComplex *B, int ldb, 
                   cuComplex beta, cuComplex *C, int ldc) 
    */
    cublasCgemm(transa, transb, m, n, k, c_alpha, A->d_mem->d_ptr, lda, B->d_mem->d_ptr, ldb, c_beta, C->d_mem->d_ptr, ldc);

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
  cuda_Array *A, *B, *C;
  Py_complex alpha, beta;
  char transa, transb;

  if (PyArg_ParseTuple(args, "ccDO!O!DO!", 
                       &transa, &transb, &alpha, 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B, 
                       &beta, cuda_ArrayType, &C)) {

    cuDoubleComplex c_alpha;
    cuDoubleComplex c_beta;

    c_alpha.x = alpha.real;
    c_alpha.y = alpha.imag;
    c_beta.x = alpha.real;
    c_beta.y = alpha.imag;

    // new setup for index twiddling transpose
    int m = transa == 't' ? A->a_dims[1] : A->a_dims[0];
    int k = transa == 't' ? A->a_dims[0] : A->a_dims[1]; 
    int kb = transb == 't' ? B->a_dims[1] : B->a_dims[0];
    int n = transb == 't' ? B->a_dims[0] : B->a_dims[1];
    int mc = C->a_dims[0];
    int nc = C->a_dims[1];

    // check geometry is good
    if (k != kb || m != mc || n != nc) {
      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-matrix mutiplication");
      return NULL;
    }

    int lda = A->a_dims[0];
    int ldb = B->a_dims[0];
    int ldc = C->a_dims[0];

    /*
      void 
      cublasZgemm (char transa, char transb, int m, int n, 
                   int k, cuComplex alpha, const cuDoubleComplex *A, 
                   int lda, const cuDoubleComplex *B, int ldb, 
                   cuDoubleComplex beta, cuDoubleComplex *C, int ldc) 
    */
    cublasZgemm(transa, transb, m, n, k, c_alpha, A->d_mem->d_ptr, lda, B->d_mem->d_ptr, ldb, c_beta, C->d_mem->d_ptr, ldc);

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
  cuda_Array *A, *B;
    
  if (PyArg_ParseTuple(args, "O!O!", 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B)) {

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

    float ip = cublasSdot(lda, A->d_mem->d_ptr, 1, B->d_mem->d_ptr, 1);

    if (cublas_error("sdot")) 
      return NULL;
    else 
      return Py_BuildValue("f", ip);
  
  } else {
    return NULL;
  }
}


/*
sgemv - single precision real matrix-vector multiply
transa, alpha, A, B, beta, C
return the updated array C 
*/

static PyObject* sgemv(PyObject* self, PyObject* args) {
  cuda_Array *A, *B, *C;
  float alpha, beta;
  char transa;

  if (PyArg_ParseTuple(args, "cfO!O!fO!", 
                       &transa, &alpha, 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B, &beta, 
                       cuda_ArrayType, &C)) {

    // new setup for index twiddling transpose
    int m = transa == 't' ? A->a_dims[1] : A->a_dims[0];
    int k = transa == 't' ? A->a_dims[0] : A->a_dims[1]; 
    int kb = B->a_dims[0];
    int n = B->a_dims[1];
    int mc = C->a_dims[0];
    int nc = C->a_dims[1];

    // check geometry is good
    if (k != kb || m != mc || n != nc) {
      PyErr_SetString(PyExc_ValueError, "matrices have wrong shapes for matrix-vector mutiplication");
      return NULL;
    }

    int lda = A->a_dims[0];

    /*
     void 
     cublasSgemv (char trans, int m, int n, float alpha, 
                  const float *A, int lda, const float *x,  
                  int incx, float beta, float *y, int incy)
    */
    cublasSgemv(transa, m, n, alpha, A->d_mem->d_ptr, lda, B->d_mem->d_ptr, 1, beta, C->d_mem->d_ptr, 1);

    if (cublas_error("sgemv")) 
      return NULL;
    else 
      return Py_BuildValue("O", C);
  
  } else {
    return NULL;
  }
}


/*
ddot - double precision real vector-vector dot product
returns a python double
*/

static PyObject* ddot(PyObject* self, PyObject* args) {
  cuda_Array *A, *B;
    
  if (PyArg_ParseTuple(args, "O!O!", 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B)) {

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

    double ip = cublasDdot(lda, A->d_mem->d_ptr, 1, B->d_mem->d_ptr, 1);

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
  cuda_Array *A, *B;
    
  if (PyArg_ParseTuple(args, "O!O!", 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B)) {

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

    cuComplex cip = cublasCdotu(lda, A->d_mem->d_ptr, 1, B->d_mem->d_ptr, 1);

    if (cublas_error("cdotu")) 
      return NULL;
    else {

      Py_complex ip;
      ip.real = (double) cip.x;
      ip.imag = (double) cip.y;
      return Py_BuildValue("D", &ip);
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
  cuda_Array *A, *B;
    
  if (PyArg_ParseTuple(args, "O!O!", 
                       cuda_ArrayType, &A, 
                       cuda_ArrayType, &B)) {

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

    cuComplex cip = cublasCdotc(lda, A->d_mem->d_ptr, 1, B->d_mem->d_ptr, 1);

    if (cublas_error("cdotc")) 
      return NULL;
    else {
      Py_complex ip;
      ip.real = (double) cip.x;
      ip.imag = (double) cip.y;
      //trace("cip: (%f,%f) -> ip: (%f,%f)\n", cip.x, cip.y, ip.real, ip.imag);
      return Py_BuildValue("D", &ip);
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

  {"sgemv", sgemv, METH_VARARGS,
   "Single Precision BLAS2 float matrix vector multiply: C = alpha * op(A) * B + beta * C"},


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

