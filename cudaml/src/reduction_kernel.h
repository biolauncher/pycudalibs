#include <cudaml.h>
#include <cudautl.h>

/************************************************
 * Whole array (vector/matrix) reduction kernels
 ************************************************/

/* Utility class used to avoid linker errors with extern unsized
   shared memory arrays of templated type */

template<class T>
struct SharedMemory {

    __device__ inline operator       T*() {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};


/*
    This kernel reduces multiple elements per thread sequentially.
    This reduces the overall cost of the algorithm while keeping the
    work complexity O(n) and the step complexity O(log n).

    N.B. needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    else if blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/

#define BINARYOP(OP, X, Y)                        \
  OP((X),(Y))

#define REDUCTION_KERNEL(NAME, BINOP)                                                                        \
                                                                                                             \
template <class T, unsigned int blockSize, bool nIsPow2>                                                     \
__global__ void                                                                                              \
NAME##_reduce_kernel(T *g_idata, T *g_odata, unsigned int n) {                                               \
                                                                                                             \
  T *sdata = SharedMemory<T>();                                                                              \
                                                                                                             \
  /* perform first level of reduction,                                                                       \
     reading from global memory, writing to shared memory */                                                 \
                                                                                                             \
  unsigned int tid = threadIdx.x;                                                                            \
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;                                                     \
  unsigned int gridSize = blockSize*2*gridDim.x;                                                             \
                                                                                                             \
  T reduction = 0;                                                                                           \
                                                                                                             \
  /* we reduce multiple elements per thread.  The number is determined by the                                \
     number of active thread blocks (via gridDim).  More blocks will result                                  \
     in a larger gridSize and therefore fewer elements per thread */                                         \
                                                                                                             \
  while (i < n) {                                                                                            \
    reduction = BINARYOP(BINOP, reduction, g_idata[i]);                                                      \
    /* ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays */             \
    if (nIsPow2 || i + blockSize < n)                                                                        \
      reduction = BINARYOP(BINOP, reduction, g_idata[i+blockSize]);                                          \
    i += gridSize;                                                                                           \
  }                                                                                                          \
                                                                                                             \
  /* each thread puts its local sum into shared memory */                                                    \
  sdata[tid] = reduction;                                                                                    \
  __syncthreads();                                                                                           \
                                                                                                             \
                                                                                                             \
  /* do reduction in shared mem */                                                                           \
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] = reduction = BINARYOP(BINOP, reduction, sdata[tid + 256]); } __syncthreads(); } \
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] = reduction = BINARYOP(BINOP, reduction, sdata[tid + 128]); } __syncthreads(); } \
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] = reduction = BINARYOP(BINOP, reduction, sdata[tid +  64]); } __syncthreads(); } \
                                                                                                             \
  if (tid < 32) {                                                                                            \
                                                                                                             \
    /* this is warp synchronous however need to ensure compiler does not reorder stores to smem */           \
    volatile T* smem = sdata;                                                                                \
                                                                                                             \
    if (blockSize >=  64) { smem[tid] = reduction = BINARYOP(BINOP, reduction, smem[tid + 32]); __syncthreads(); } \
    if (blockSize >=  32) { smem[tid] = reduction = BINARYOP(BINOP, reduction, smem[tid + 16]); __syncthreads(); } \
    if (blockSize >=  16) { smem[tid] = reduction = BINARYOP(BINOP, reduction, smem[tid +  8]); __syncthreads(); } \
    if (blockSize >=   8) { smem[tid] = reduction = BINARYOP(BINOP, reduction, smem[tid +  4]); __syncthreads(); } \
    if (blockSize >=   4) { smem[tid] = reduction = BINARYOP(BINOP, reduction, smem[tid +  2]); __syncthreads(); } \
    if (blockSize >=   2) { smem[tid] = reduction = BINARYOP(BINOP, reduction, smem[tid +  1]); __syncthreads(); } \
  }                                                                                                          \
                                                                                                             \
  /* write result for this block to global mem */                                                            \
  if (tid == 0)                                                                                              \
    g_odata[blockIdx.x] = sdata[0];                                                                          \
}


/* define kernel launcher template for a given operator */

#define REDUCTION_KERNEL_LAUNCHER(O)                                                                         \
                                                                                                             \
template <class T>                                                                                           \
void                                                                                                         \
O##_reduce(int size, int threads, int blocks, T *d_idata, T *d_odata) {                                      \
  dim3 dimBlock(threads, 1, 1);                                                                              \
  dim3 dimGrid(blocks, 1, 1);                                                                                \
                                                                                                             \
  /* when there is only one warp per block, we need to allocate two warps                                    \
     worth of shared memory so that we don't index shared memory out of bounds */                            \
                                                                                                             \
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);                            \
                                                                                                             \
  if (isPow2(size)) {                                                                                        \
    switch (threads) {                                                                                       \
    case 512:                                                                                                \
      O##_reduce_kernel<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case 256:                                                                                                \
      O##_reduce_kernel<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case 128:                                                                                                \
      O##_reduce_kernel<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case 64:                                                                                                 \
      O##_reduce_kernel<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case 32:                                                                                                 \
      O##_reduce_kernel<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case 16:                                                                                                 \
      O##_reduce_kernel<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case  8:                                                                                                 \
      O##_reduce_kernel<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case  4:                                                                                                 \
      O##_reduce_kernel<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case  2:                                                                                                 \
      O##_reduce_kernel<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    case  1:                                                                                                 \
      O##_reduce_kernel<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;     \
    }                                                                                                        \
  } else {                                                                                                   \
    switch (threads) {                                                                                       \
    case 512:                                                                                                \
      O##_reduce_kernel<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case 256:                                                                                                \
      O##_reduce_kernel<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case 128:                                                                                                \
      O##_reduce_kernel<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case 64:                                                                                                 \
      O##_reduce_kernel<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case 32:                                                                                                 \
      O##_reduce_kernel<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case 16:                                                                                                 \
      O##_reduce_kernel<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case  8:                                                                                                 \
      O##_reduce_kernel<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case  4:                                                                                                 \
      O##_reduce_kernel<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case  2:                                                                                                 \
      O##_reduce_kernel<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    case  1:                                                                                                 \
      O##_reduce_kernel<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;    \
    }                                                                                                        \
  }                                                                                                          \
}


#define ARRAY_REDUCTION_KERNEL_LAUNCHER(OP, TYPE)                                                           \
template void                                                                                               \
OP##_reduce<TYPE>(int size, int threads, int blocks, TYPE* d_idata, TYPE* d_odata);


#define ARRAY_REDUCE(OP, TYPE)                                                                              \
                                                                                                            \
/* API function */                                                                                          \
cudaError_t cudaml_a##OP(TYPE* A, size_t size, TYPE* res) {                                                 \
                                                                                                            \
  /* calculate block and grid sizes based on input job size */                                              \
  int threads = rk_threads(size);                                                                           \
  int blocks = rk_blocks(size, threads);                                                                    \
  int n = (int) size;                                                                                       \
                                                                                                            \
  trace("cudaml_a## n: [%d], threads: %d, blocks: %d\n", n, threads, blocks);                               \
                                                                                                            \
  cudaError_t sts;                                                                                          \
  TYPE* C;                                                                                                  \
  /* allocate output data vector - which is number of blocks (since this is first pass reduction size) */   \
  if ((sts = cudaMalloc(&C, blocks * sizeof(TYPE))) != cudaSuccess)                                         \
    return sts;                                                                                             \
                                                                                                            \
  /* fire up the kernel to compute the reduction on GPU */                                                  \
  OP##_reduce<TYPE>(n, threads, blocks, A, C);                                                              \
                                                                                                            \
  trace("cudaml_a##_reduct1: %d\n", blocks);                                                                \
                                                                                                            \
  /* C should contain "blocks" partial sums so we reduce iteratively again in place                         \
     and sum partial block sums on GPU  - this might be inefficient for short vectors                       \
     of partial sums so we could set a threshold for this and do final reduct on CPU */                     \
                                                                                                            \
  int s = blocks;                                                                                           \
                                                                                                            \
  while (s > 1) {                                                                                           \
    int nt = rk_threads(s);                                                                                 \
    int nb = rk_blocks(s, nt);                                                                              \
                                                                                                            \
    OP##_reduce<TYPE>(s, nt, nb, C, C);                                                                     \
    trace("cudaml_a##_reduct+: %d\n", s);                                                                   \
    s = nb;                                                                                                 \
  }                                                                                                         \
                                                                                                            \
  /* C[0] should be our result - so we need to copy the data back to host */                                \
  cudaMemcpy(res, C, sizeof(TYPE), cudaMemcpyDeviceToHost);                                                 \
  return cudaFree(C);                                                                                       \
}


#define COLUMN_REDUCE(OP,TYPE)                                                                              \
                                                                                                            \
/* API template reduce m rows of T in to n columns a and r should be                                        \
   device memory */                                                                                         \
cudaError_t cudaml_c##OP(TYPE* a, size_t m, size_t n, TYPE* r) {                                            \
                                                                                                            \
  /* per column/dimension reduction - we invoke kernel n times with m rows */                               \
  int threads = rk_threads(m);                                                                              \
  int blocks = rk_blocks(m, threads);                                                                       \
                                                                                                            \
  trace("cudaml_c##: [%d,%d], threads: %d, blocks: %d\n", (int) m, (int) n, threads, blocks); \
                                                                                                            \
  float* c;                                                                                                 \
  cudaError_t sts;                                                                                          \
                                                                                                            \
  /* allocate enough device memory for a block reduction */                                                 \
  if ((sts = cudaMalloc(&c, blocks * sizeof(TYPE))) != cudaSuccess)                                         \
    return sts;                                                                                             \
                                                                                                            \
  for (int j=0; j<n; j++) {                                                                                 \
                                                                                                            \
    float* A = a+m*j;                                                                                       \
                                                                                                            \
    OP##_reduce<TYPE>(m, threads, blocks, A, c);                                                            \
                                                                                                            \
    trace("cudaml_c##_reduct1: %d\n", blocks);                                                              \
                                                                                                            \
    int s = blocks;                                                                                         \
                                                                                                            \
    while (s > 1) {                                                                                         \
      int nt = rk_threads(s);                                                                               \
      int nb = rk_blocks(s, nt);                                                                            \
                                                                                                            \
      OP##_reduce<TYPE>(s, nt, nb, c, c);                                                                   \
      trace("cudaml_c##_reduct+: %d\n", s);                                                                 \
      s = nb;                                                                                               \
    }                                                                                                       \
                                                                                                            \
    cudaMemcpy(r+j, c, sizeof(float), cudaMemcpyDeviceToDevice);                                            \
  }                                                                                                         \
  cudaFree(c);                                                                                              \
  return cudaGetLastError();                                                                                \
}

