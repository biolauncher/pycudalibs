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
  
  // real kernels 
  
  static float sum;
  static float A[] = {1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.};
  static float R[] = {0,0,0,0,0,0,0,0,0,0,0,0};
  static float P[] = {1.,2.,5.,7.,2};

  static float* C;
  static float* V;
  static float* D;

  trace("test_main: size: %ld\n", sizeof(A));
  cuda_error(cudaMalloc((void**) &C, sizeof(A)), "device array allocation");
  cuda_error(cudaMemcpy((void *)C, A, sizeof(A), cudaMemcpyHostToDevice), "host to device copy");

  cuda_error(cudaml_asum(C, sizeof(A)/sizeof(float), &sum), "cudaml_asum invokation");
  trace("test_main: result: %f\n", sum);

  cuda_error(cudaMalloc((void**) &V, sizeof(float) * 3), "device vector allocation");  
  cuda_error(cudaml_csum(C, 4, 3, V), "cudaml_csum invokation");
  cuda_error(cudaMemcpy((void*) R, V, sizeof(float) * 3, cudaMemcpyDeviceToHost), "vector device to host copy");
  cudaFree(V);

  cuda_error(cudaMalloc((void**) &V, sizeof(A)), "device array allocation");  
  cuda_error(cudaml_esum(C, sizeof(A), 27.36, V), "cudaml_esum invokation");
  cuda_error(cudaMemcpy((void*) R, V, sizeof(A), cudaMemcpyDeviceToHost), "array device to host copy");
  cudaFree(V);

  cuda_error(cudaMalloc((void**) &V, sizeof(A)), "device array allocation");  
  cuda_error(cudaMalloc((void**) &D, sizeof(P)), "device vector allocation");  
  cuda_error(cudaMemcpy((void *)D, P, sizeof(P), cudaMemcpyHostToDevice), "host to device copy");

  cuda_error(cudaml_evsum(C, sizeof(A), D, sizeof(P)/sizeof(float), V), "cudaml_evsum invokation");
  cuda_error(cudaMemcpy((void*) R, V, sizeof(A), cudaMemcpyDeviceToHost), "array device to host copy");
  cudaFree(D);
  cudaFree(V);


  cuda_error(cudaDeviceReset(), "device reset");
}
