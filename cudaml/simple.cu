/*
* simple kernel for testing
*/
#include <cuda_runtime_api.h>
#include <cudaml.h>

__global__ void simple_kernel(void* v, size_t s) { }

 
cudaError_t simple(void* v, size_t s) {
  // TODO allocate some device memory and call kernel
  simple_kernel<<<1,1>>> (v, s);
  return cudaGetLastError();
}

