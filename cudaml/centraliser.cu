#include <cudaml.h>

/* device kernel */
__global__ void centraliser_kernel(void* a, void* c, size_t m, size_t n) { 

}

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

/* host function */
cudaError_t cudaml_centraliser(void* a, void*c, size_t m, size_t n) {
  
  dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
  dim3 gridDim(n/blockDim.x+1, m/blockDim.y+1);

  centraliser_kernel<<<gridDim, blockDim>>>(a, c, m, n);
  return cudaGetLastError();
}
