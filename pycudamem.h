/* copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/* this is the master header that should be included my all c extension modules here */

#if defined(_PYCUDA_H)
#else
#define _PYCUDA_H 1
#include <Python.h>
#include <cublas.h>

#if DEBUG > 0
#include <stdio.h>
#define trace(format, ...) fprintf(stderr, format, ## __VA_ARGS__)
#else
#define trace(format, ...)
#endif

/* A python base class to encapsulate CUDA device memory. */

typedef struct {
  PyObject_HEAD
  void* d_ptr; // don't even think about de-referencing this!
  int e_size;
  int e_num;
} cuda_DeviceMemory;

#define FLOAT32_BYTES 4

#if defined(CUDAMEM_MODULE)

/* Lookup of cublas error since we use CUBLAS routines in _cudamem module for allocation */

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
#else

/* client code */
static PyTypeObject *cudamem_DeviceMemoryType;
 
static inline int import_cudamem(void) {
  PyObject *module = PyImport_ImportModule("_cudamem");
  
  if (module != NULL) {
    cudamem_DeviceMemoryType = (PyTypeObject *)PyObject_GetAttrString(module, "DeviceMemory");
    if (cudamem_DeviceMemoryType == NULL) return -1;

  } 
  return 0;
}

#endif /* client code */

#endif


