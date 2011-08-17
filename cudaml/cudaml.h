/* cudaml library header */ 
#ifndef __CUDAML_H__
#define __CUDAML_H__ 1

#ifdef __cplusplus
#include <cuda_runtime.h>
extern "C" {
#else
#include <cuda_runtime_api.h>
#endif
  // Library functions
  cudaError_t null(void);
  cudaError_t simple(void*, size_t);
  cudaError_t cudaml_centraliser(void*, void*, size_t, size_t);
  cudaError_t sum(float*, size_t, float*);
#ifdef __cplusplus
}
#endif

#endif // __CUDAML_H__
