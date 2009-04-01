/* copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/* 
 * This is the master header that should be included in all c extension modules in the package 
 * it defines the _cudamem module which provides a cuda array type along the lines of numpy
 */

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


/* defines for various things */

#define CUDAMEM_MODULE_NAME "_cudamem"
#define CUDAMEM_ARRAY_TYPE_NAME "_cudamem.array"
#define CUDAMEM_ERROR_TYPE_NAME "_cudamem.ERROR"

#define DEVICE_MEMORY_MAXDIMS 2 
#define FLOAT32_BYTES 4
#define FLOAT64_BYTES 8
#define COMPLEX32_BYTES 8
#define COMPLEX64_BYTES 16

#define COMPLEX_TYPE 1
#define DOUBLE_TYPE 2

/* A python base class to encapsulate CUDA device memory. Fortran array semantics apply.  */

typedef struct {
  PyObject_HEAD
  void* d_ptr;                       /* opaque device pointer */
  int a_ndims;                       /* number of dimensions */
  int a_dims[DEVICE_MEMORY_MAXDIMS]; /* dimensions */
  int e_size;                        /* sizeof element */
  int a_flags;
} cuda_DeviceMemory;

/* return number of elements: matrix, vector or scalar */

static inline int a_elements(cuda_DeviceMemory* d) {
  return (d->a_ndims == 2) ? (d->a_dims[0] * d->a_dims[1]) : (d->a_ndims == 1 ? d->a_dims[0] : 1);
}

/* module based exception */
static PyObject* cuda_exception;

/* Lookup of cublas error since we use CUBLAS routines in _cudamem module for allocation */

static char* cublas_error_text [] = {
  "CUBLAS library not initialized",
  "Resource allocation failed",
  "Unsupported numerical value was passed to function",
  "Function requires a feature absent from the architecture of the device",
  "Access to GPU memory space failed",
  "GPU program failed to execute",
  "An internal CUBLAS operation failed"};

#if defined(CUDAMEM_MODULE)
/* module definition only - see: pycudamem.c */

#else

/* client code only */

static PyTypeObject *cudamem_DeviceMemoryType;
 
static inline int import_cudamem(void) {
  PyObject *module = PyImport_ImportModule(CUDAMEM_MODULE_NAME);
  
  if (module != NULL) {
    cudamem_DeviceMemoryType = (PyTypeObject *)PyObject_GetAttrString(module, CUDAMEM_ARRAY_TYPE_NAME);
    if (cudamem_DeviceMemoryType == NULL) return -1;

    cuda_exception = PyObject_GetAttrString(module, CUDAMEM_ERROR_TYPE_NAME);
    if (cuda_exception == NULL) return -1;
  } 
  return 0;
}

#endif /* client code */

/* inline utilities */

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

static inline int cuda_error(int status, char* where) {
  trace("CUDACALL %s: status = %d\n", where, status);

  if (status == CUBLAS_STATUS_SUCCESS) {
    return 0;

  } else {
    PyErr_SetString(cuda_exception, get_cublas_error_text(status));
    return 1;
  }
}

static inline int cublas_error(char* where) {
  return cuda_error(cublasGetError(), where);
}

#endif


