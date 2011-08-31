/* cudaml library header */ 
#ifndef __CUDAML_H__
#define __CUDAML_H__ 1

#define ARRAY_REDUCE_PROTO(NAME, TYPE) cudaError_t cudaml_a##NAME(TYPE*, size_t, TYPE*);

#ifdef __cplusplus
#include <cuda_runtime.h>
extern "C" {
#else
#include <cuda_runtime_api.h>
#endif
  // Library functions
  cudaError_t null(void);
  // device memory functions with host scalar reduction
  cudaError_t cudaml_asum(float*, size_t, float*);
  cudaError_t cudaml_amax(float*, size_t, float*);
  cudaError_t cudaml_amin(float*, size_t, float*);
  cudaError_t cudaml_aproduct(float*, size_t, float*);
  // device memory functions with device vector reduction
  cudaError_t cudaml_csum(float*, size_t, size_t, float*);
  cudaError_t cudaml_cmax(float*, size_t, size_t, float*);
  cudaError_t cudaml_cmin(float*, size_t, size_t, float*);
  cudaError_t cudaml_cproduct(float*, size_t, size_t, float*);
  // device memory functions withelement wise operations
  cudaError_t cudaml_esum(float*, size_t, float, float*);
  cudaError_t cudaml_evsum(float*, size_t, float*, size_t, float*);
  cudaError_t cudaml_emul(float*, size_t, float, float*);
  cudaError_t cudaml_evmul(float*, size_t, float*, size_t, float*);
  cudaError_t cudaml_easum(float*, size_t, size_t, float*, size_t, float*);
  cudaError_t cudaml_eamul(float*, size_t, size_t, float*, size_t, float*);
  // device memory functions with device matrix result
  // this in python for now cudaError_t cudaml_centraliser(void*, void*, size_t, size_t);

#ifdef __cplusplus
}
#endif

#endif // __CUDAML_H__
