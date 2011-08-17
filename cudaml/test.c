#include <stdio.h>
#include <cudaml.h>

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
  cuda_error(cudaml_centraliser(NULL, NULL, 0, 0), "cudaml_centraliser invokation");
}
