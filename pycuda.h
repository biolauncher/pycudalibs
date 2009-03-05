/* copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/* this is the master header that should be included my all c extension modules here */

#if defined(_PYCUDA_H)
#else
#define _PYCUDA_H 1
#include <cublas.h>
#include <Python.h>

/* CUDA utilities */

/* A python base class to encapsulate CUDA device memory. */

typedef struct {
  PyObject_HEAD
  void* d_ptr; // don't even think about de-referencing this!
  int e_size;
  int e_num;
} cuda_DeviceMemory;


/* Lookup of cublas error since we use CUBLAS routines in _cuda module for allocation */

static char* cublas_error_text [] = {
  "CUBLAS library not initialized",
  "Resource allocation failed",
  "Unsupported numerical value was passed to function",
  "Function requires a feature absent from the architecture of the device",
  "Access to GPU memory space failed",
  "GPU program failed to execute",
  "An internal CUBLAS operation failed"};

static inline char* get_cublas_error_text(cublasStatus sts) {
  switch (sts) {
    case CUBLAS_STATUS_NOT_INITIALIZED :
      return cublas_error_text[0];
    case CUBLAS_STATUS_ALLOC_FAILED :
      return cublas_error_text[1];
    case CUBLAS_STATUS_INVALID_VALUE :
      return cublas_error_text[2];
    case CUBLAS_STATUS_ARCH_MISMATCH :
      return cublas_error_text[3];
    case CUBLAS_STATUS_MAPPING_ERROR :
      return cublas_error_text[4];
    case CUBLAS_STATUS_EXECUTION_FAILED :
      return cublas_error_text[5];
    case CUBLAS_STATUS_INTERNAL_ERROR :
      return cublas_error_text[6];
    default:
      return "unknown cublas error!";
  }
}

#endif


