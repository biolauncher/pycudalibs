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
  // library functions
  cudaError_t null(void);
  
  // device memory functions with reduction to host scalar
  cudaError_t cudaml_asum(float*, size_t, float*);
  cudaError_t cudaml_amax(float*, size_t, float*);
  cudaError_t cudaml_amin(float*, size_t, float*);
  cudaError_t cudaml_aproduct(float*, size_t, float*);
  
  // device memory functions with device vector reduction
  cudaError_t cudaml_csum(float*, size_t, size_t, float*);
  cudaError_t cudaml_cmax(float*, size_t, size_t, float*);
  cudaError_t cudaml_cmin(float*, size_t, size_t, float*);
  cudaError_t cudaml_cproduct(float*, size_t, size_t, float*);
  
  // device memory functions with element wise operations
  cudaError_t cudaml_esum(float*, size_t, float, float*);
  cudaError_t cudaml_evsum(float*, size_t, float*, size_t, float*);
  cudaError_t cudaml_easum(float*, size_t, size_t, float*, size_t, float*);

  cudaError_t cudaml_emul(float*, size_t, float, float*);
  cudaError_t cudaml_evmul(float*, size_t, float*, size_t, float*);
  cudaError_t cudaml_eamul(float*, size_t, size_t, float*, size_t, float*);

  cudaError_t cudaml_epow(float*, size_t, float, float*);
  cudaError_t cudaml_evpow(float*, size_t, float*, size_t, float*);
  cudaError_t cudaml_eapow(float*, size_t, size_t, float*, size_t, float*);
 
  // unary goodness
  cudaError_t cudaml_usqrt(float*, size_t, float*);
  cudaError_t cudaml_ulog(float*, size_t, float*);
  cudaError_t cudaml_ulog2(float*, size_t, float*);
  cudaError_t cudaml_ulog10(float*, size_t, float*);
  
  cudaError_t cudaml_usin(float*, size_t, float*);
  cudaError_t cudaml_ucos(float*, size_t, float*);
  cudaError_t cudaml_utan(float*, size_t, float*);
  
  cudaError_t cudaml_usinh(float*, size_t, float*);
  cudaError_t cudaml_ucosh(float*, size_t, float*);
  cudaError_t cudaml_utanh(float*, size_t, float*);
  
  cudaError_t cudaml_uexp(float*, size_t, float*);
  cudaError_t cudaml_uexp10(float*, size_t, float*);
  
  cudaError_t cudaml_usinpi(float*, size_t, float*);
  cudaError_t cudaml_ucospi(float*, size_t, float*);
  
  cudaError_t cudaml_uasin(float*, size_t, float*);
  cudaError_t cudaml_uacos(float*, size_t, float*);
  cudaError_t cudaml_uatan(float*, size_t, float*);
  cudaError_t cudaml_uasinh(float*, size_t, float*);
  cudaError_t cudaml_uacosh(float*, size_t, float*);
  cudaError_t cudaml_uatanh(float*, size_t, float*);
  
  cudaError_t cudaml_uerf(float*, size_t, float*);
  cudaError_t cudaml_uerfc(float*, size_t, float*);
  cudaError_t cudaml_uerfinv(float*, size_t, float*);
  cudaError_t cudaml_uerfcinv(float*, size_t, float*);
  cudaError_t cudaml_ulgamma(float*, size_t, float*);
  cudaError_t cudaml_utgamma(float*, size_t, float*);

  cudaError_t cudaml_utrunc(float*, size_t, float*);
  cudaError_t cudaml_uround(float*, size_t, float*);
  cudaError_t cudaml_urint(float*, size_t, float*);
  cudaError_t cudaml_ufloor(float*, size_t, float*);
  cudaError_t cudaml_uceil(float*, size_t, float*);

  // device memory functions with device matrix result
  // this in python for now cudaError_t cudaml_centraliser(void*, void*, size_t, size_t);

#ifdef __cplusplus
}
#endif

#endif // __CUDAML_H__
