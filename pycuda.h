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

#if defined(_PYCUDA_H)
#else
#define _PYCUDA_H 1

#include <cublas.h>
#include <pycunumpy.h>
#include <pylibs.h>

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

/* CUDA error handling XXX deprecated rename as cudablas_error */
static inline int cuda_error(int status, char* where) {
  trace("CUDACALL %s: status = %d\n", where, status);

  if (status == CUBLAS_STATUS_SUCCESS) {
    return 0;

  } else {
    
    return 1;
  }
}

/* new CUDA error handling */
static inline cuda_error2(cudaError_t sts, char* info) {
  if (sts != cudaSuccess) {
    const char* error_text = cudaGetErrorString(sts);
    PyErr_SetString(cuda_exception, error_text);

    trace("CUDA_EXCEPTION\t %s\t(%d)\t%s\n", info, sts, error_text);
    return 1;

  } else {
    trace("CUDA_SUCCESS\t %s\n", info);
    return 0;
  }
}

/* cublas error */
static inline int cublas_error(char* where) {
  return cuda_error(cublasGetError(), where);
}

#endif
