#include <cudaml.h>
#include <cudautl.h>

/* Utility class used to avoid linker errors with extern unsized
   shared memory arrays with templated type */

template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};



/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce_kernel(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    /* perform first level of reduction,
       reading from global memory, writing to shared memory */

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum = 0;

    /* we reduce multiple elements per thread.  The number is determined by the 
       number of active thread blocks (via gridDim).  More blocks will result
       in a larger gridSize and therefore fewer elements per thread */

    while (i < n) {         
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      if (nIsPow2 || i + blockSize < n) 
        mySum += g_idata[i+blockSize];  
      i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32) {
      /* now that we are using warp-synchronous programming (below)
         we need to declare our shared memory volatile so that the compiler
         doesn't reorder stores to it and induce incorrect behavior. */

      volatile T* smem = sdata;

      if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; __syncthreads(); }
      if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; __syncthreads(); }
      if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; __syncthreads(); }
      if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; __syncthreads(); }
      if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; __syncthreads(); }
      if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; __syncthreads(); }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
      g_odata[blockIdx.x] = sdata[0];
}


/* wrapper function for kernel launch */

template <class T>
void 
reduce(int size, int threads, int blocks, T *d_idata, T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  /* when there is only one warp per block, we need to allocate two warps 
     worth of shared memory so that we don't index shared memory out of bounds */

  int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  if (isPow2(size)) {
    switch (threads) {
    case 512:
      reduce_kernel<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
      reduce_kernel<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
      reduce_kernel<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
      reduce_kernel<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 32:
      reduce_kernel<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 16:
      reduce_kernel<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  8:
      reduce_kernel<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  4:
      reduce_kernel<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  2:
      reduce_kernel<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  1:
      reduce_kernel<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  } else {
    switch (threads) {
    case 512:
      reduce_kernel<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 256:
      reduce_kernel<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 128:
      reduce_kernel<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 64:
      reduce_kernel<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 32:
      reduce_kernel<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case 16:
      reduce_kernel<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  8:
      reduce_kernel<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  4:
      reduce_kernel<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  2:
      reduce_kernel<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    case  1:
      reduce_kernel<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
    }
  }
}

// generate the reduction function for float
template void 
reduce<float>(int size, int threads, int blocks, float* d_idata, float* d_odata);

/* actual host function that does the job */
cudaError_t sum(float* A, size_t size, float* sum) {

  // calculate block and grid sizes based on input job size
  int threads = rk_threads(size);
  int blocks = rk_blocks(size, threads);
  int n = (int) size;

  cudaError_t sts;
  float* C;
  // allocate output data vector - which is number of blocks (since this is first pass reduction size)
  if ((sts = cudaMalloc(&C, blocks * sizeof(float))) != cudaSuccess)
    return sts;

  // fire up the kernel to compute the reduction on GPU
  reduce<float>(n, threads, blocks, A, C);

  /* C should contain "blocks" partial sums so we reduce iteratively again in place
     and sum partial block sums on GPU  - this might be inefficient for short vectors
     of partial sums so we could set a threshold for this and do final reduct on CPU */

  int s = blocks;

  while (s > 1) {
    int nt = rk_threads(s);
    int nb = rk_blocks(s, nt);
                
    reduce<float>(s, nt, nb, C, C);                
    s = nb;
  }         
  
  // C[0] should be our result - so we need to copy the data back!
  cudaMemcpy(sum, C, sizeof(float), cudaMemcpyDeviceToHost);
  // free the device vector C
  return cudaFree(C);  
}
