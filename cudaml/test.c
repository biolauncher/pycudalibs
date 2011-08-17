#include <stdio.h>
#include <cudaml.h>
#include <cudautl.h>

static inline int cuda_error(cudaError_t sts, const char* info) {

  if (sts != cudaSuccess) {
    const char* error_text = cudaGetErrorString(sts);
    printf("CUDA_EXCEPTION: \t%s (%d) %s\n", info, sts, error_text);
    return 1;

  } else {
    printf("CUDA_SUCCESS: \t%s\n", info) ;
    return 0;
  }
}


/* test stubs */

int main(int argc, char** argv) {
  cuda_error(null(), "null kernel invokation");
  cuda_error(simple(NULL, 0), "simple kernel invokation");
  
  // real kernels
  cuda_error(cudaml_centraliser(NULL, NULL, 0, 0), "cudaml_centraliser invokation");


  
  static float sum;
  static float A[] = {1.,2.,3.,4.,5.,6.,7.,8.,9.};
  static float* C;
  trace("size: %ld\n", sizeof(A));
  cuda_error(cudaMalloc((void**) &C, sizeof(A)), "device array allocation");
  cuda_error(cudaMemcpy((void *)C, A, sizeof(A), cudaMemcpyHostToDevice), "host to device copy");
  cuda_error(cudaml_asum(C, sizeof(A)/sizeof(float), &sum), "cudaml_asum invokation");

  cuda_error(cudaDeviceReset(), "device reset");
}
