/* copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/* 
 * This is the master header that should be included in all c extension modules in the package 
 * it defines the _cunumpy module which provides a cuda array type along the lines of numpy
 */

#if defined(_PYCUNUMPY_H)
#else
#define _PYCUNUMPY_H 1
#include <Python.h>
#include <cublas.h>
#include <structmember.h>
#include <arrayobject.h>

#if DEBUG > 0
#include <stdio.h>
#define trace(format, ...) fprintf(stderr, format, ## __VA_ARGS__)
#else
#define trace(format, ...)
#endif


/* defines for various things */

#define CUDA_MODULE_NAME "_cunumpy"

#define CUDA_ARRAY_TYPE_NAME "cuda.array"
#define CUDA_ARRAY_TYPE_SYM_NAME "array"

#define CUDA_ERROR_TYPE_NAME "cuda.CUDAERROR"
#define CUDA_ERROR_TYPE_SYM_NAME "CUDAERROR"


#define DEVICE_ARRAY_MAXDIMS 2 
#define DEVICE_ARRAY_MINDIMS 1

#define DEVICE_ARRAY_TYPE_OK(dt) \
  ((PyTypeNum_ISCOMPLEX((dt)->type_num) && ((dt)->elsize == 8 || (dt)->elsize == 16)) \
   || (PyTypeNum_ISFLOAT((dt)->type_num) && ((dt)->elsize == 4 || (dt)->elsize == 8)))

/* A python base class to encapsulate CUDA device memory. Fortran array semantics apply.  */

typedef struct {
  PyObject_HEAD
  void* d_ptr;                       /* opaque device pointer */
  int a_ndims;                       /* number of dimensions */
  int a_dims[DEVICE_ARRAY_MAXDIMS];  /* dimensions */
  int e_size;                        /* sizeof element */
  PyArray_Descr* a_dtype;            /* keep reference to numpy dtype */
  int a_transposed;                  /* this reserved for future use */
} cuda_DeviceMemory;

/* return number of elements: matrix, vector or scalar */

static inline int a_elements(cuda_DeviceMemory* d) {
  return (d->a_ndims == 2) ? (d->a_dims[0] * d->a_dims[1]) : (d->a_ndims == 1 ? d->a_dims[0] : 1);
}

/* module based exception */
static PyObject* cuda_exception;

/* lookup of cublas error text */

static char* cublas_error_text [] = {
  "CUBLAS library not initialized",
  "Resource allocation failed",
  "Unsupported numerical value was passed to function",
  "Function requires a feature absent from the architecture of the device",
  "Access to GPU memory space failed",
  "GPU program failed to execute",
  "An internal CUBLAS operation failed"};

#if defined(CUNUMPY_MODULE)
/* module definition only - see: pycunumpy.c */

#else

/* client code only */

static PyTypeObject *cuda_DeviceMemoryType;
 
static inline int import_cunumpy(void) {
  PyObject *module = PyImport_ImportModule(CUDA_MODULE_NAME);
  
  if (module != NULL) {
    cuda_DeviceMemoryType = (PyTypeObject *)PyObject_GetAttrString(module, CUDA_ARRAY_TYPE_SYM_NAME);
    if (cuda_DeviceMemoryType == NULL) return -1;

    cuda_exception = PyObject_GetAttrString(module, CUDA_ERROR_TYPE_SYM_NAME);
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


