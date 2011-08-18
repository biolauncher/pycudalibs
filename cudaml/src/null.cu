/*
* Null kernel to get library started
*/
#include <cuda_runtime_api.h>
#include <cudaml.h>

__global__ void null_kernel(void) { }

 
cudaError_t null(void) {
  null_kernel<<<1,1>>> ();
  return cudaGetLastError();
}


