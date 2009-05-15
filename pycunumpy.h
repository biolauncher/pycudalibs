/* copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/* 
 * This is the master header that should be included in all c extension modules in the package 
 * it defines the _cunumpy module which provides a cuda array type along the lines of numpy
 * it is the header file from hell... 
 */

#if defined(_PYCUNUMPY_H)
#else
#define _PYCUNUMPY_H 1

#if DEBUG > 0
#include <stdio.h>
#define trace(format, ...) fprintf(stderr, format, ## __VA_ARGS__)
#else
#define trace(format, ...)
#endif

#include <cublas.h>

/* lookup of cublas error text */

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

#include <Python.h>

/* module based exception */
static PyObject* cuda_exception;

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

#include <structmember.h>
#include <arrayobject.h>
#include <pycumem.h>


/* defines for various constants */

#define CUDA_MODULE_NAME "_cunumpy"

#define CUDA_ARRAY_TYPE_NAME "cuda.array"
#define CUDA_ARRAY_TYPE_SYM_NAME "array"

#define CUDA_ERROR_TYPE_NAME "cuda.CUDAERROR"
#define CUDA_ERROR_TYPE_SYM_NAME "CUDAERROR"

#define CUDA_MEMORY_TYPE_NAME "cuda.memory"

#define DEVICE_ARRAY_MAXDIMS 2 
#define DEVICE_ARRAY_MINDIMS 1

#define DEVICE_ARRAY_TYPE_OK(dt) \
  ((PyTypeNum_ISCOMPLEX((dt)->type_num) && ((dt)->elsize == 8 || (dt)->elsize == 16)) \
   || (PyTypeNum_ISFLOAT((dt)->type_num) && ((dt)->elsize == 4 || (dt)->elsize == 8)))


/** 
 * A python base class to encapsulate a CUDA memory backed array
 * numpy is our inspiration and guiding light.
 * Fortran array semantics apply.
 * Vectors are conventionally column vectors.
 */

typedef struct {
  PyObject_HEAD
  cuda_Memory* d_mem;
  int a_ndims;                       /* number of dimensions */
  int a_dims[DEVICE_ARRAY_MAXDIMS];  /* dimensions */
  int e_size;                        /* sizeof element */
  PyArray_Descr* a_dtype;            /* keep reference to numpy dtype */
  int a_transposed;                  /* is array transposed */
} cuda_Array;


#define OP(A) ((A)->a_transposed ? 't' : 'n')

static inline int isvector(cuda_Array* d) {
  return d->a_ndims == 1 ? 1 : 0;
}

static inline int iscomplex(cuda_Array* d) {
  return PyTypeNum_ISCOMPLEX(d->a_dtype->type_num);
}

static inline int isdouble(cuda_Array* d) {
  return iscomplex(d) ? d->e_size == 16 : d->e_size == 8;
}

/* return number of elements: matrix, vector or scalar */

static inline int a_elements(cuda_Array* d) {
  return (d->a_ndims == 2) ? (d->a_dims[0] * d->a_dims[1]) : (d->a_ndims == 1 ? d->a_dims[0] : 1);
}


#if defined(CUNUMPY_MODULE)
/* module definition only - see: pycunumpy.c */

/* static prototypes */
static inline cuda_Array* make_vector(int, PyArray_Descr*);
static inline cuda_Array* make_matrix(int, int, PyArray_Descr*);
static inline cuda_Array* copy_array(cuda_Array*);

// TODO declare all methods in here - 
static PyObject* cuda_Array_dot(cuda_Array*, PyObject*);
static PyObject* cuda_Array_transpose(cuda_Array*, PyObject*);
static PyObject* cuda_Array_scale(cuda_Array*, PyObject*);
static PyObject* cuda_Array_copy(cuda_Array*);
static PyObject* cuda_Array_2norm(cuda_Array*);
static PyObject* cuda_Array_asum(cuda_Array*);
static PyObject* cuda_Array_reshape(cuda_Array*, PyObject*);


#else

/* client code only */

static PyTypeObject *cuda_ArrayType;
 
static inline int import_cunumpy(void) {
  PyObject *module = PyImport_ImportModule(CUDA_MODULE_NAME);
  
  if (module != NULL) {
    cuda_ArrayType = (PyTypeObject *)PyObject_GetAttrString(module, CUDA_ARRAY_TYPE_SYM_NAME);
    if (cuda_ArrayType == NULL) return -1;

    cuda_exception = PyObject_GetAttrString(module, CUDA_ERROR_TYPE_SYM_NAME);
    if (cuda_exception == NULL) return -1;
  } 
  return 0;
}

#endif /* client code */

#endif


