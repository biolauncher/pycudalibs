#include <cudaml.h>
#include <cudautl.h>

/* simple element kernel */

#define ELEMENT_KERNEL(NAME, OP)                                             \
template <class T>                                                           \
__global__ void                                                              \
NAME##_element_kernel(T *g_idata, T scalar, T* g_odata, unsigned int n) {    \
  int i = blockDim.x * blockIdx.x + threadIdx.x;                             \
  if (i < n) g_odata[i] = BINARYOP(OP, g_idata[i], scalar);                  \
}

#define ELEMENT_KERNEL_IMPL(NAME, TYPE)                                      \
template void                                                                \
NAME##_element_kernel<TYPE>(TYPE* a, TYPE scalar, TYPE* r, unsigned int n);


#define ARRAY_ELEMENTWISE(NAME, TYPE)                                        \
cudaError_t cudaml_e##NAME(TYPE* a, size_t n, TYPE s, TYPE* r) {             \
                                                                             \
 int threadsPerBlock = 256;                                                  \
 int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;            \
                                                                             \
 NAME##_element_kernel<TYPE><<<blocksPerGrid, threadsPerBlock>>>(a, s, r, n);\
 return cudaGetLastError();                                                  \
}


/* matrix element kernel m, with vector n more general than above */
// TODO - per column vector? 

#define ELEMENT_V_KERNEL(NAME, OP)                                            \
template <class T>                                                           \
__global__ void                                                              \
 NAME##_element_v_kernel(T *g_idata, T* g_ivec, T* g_odata, unsigned int m, unsigned int n) { \
  int i = blockDim.x * blockIdx.x + threadIdx.x;                             \
  int j = i;                                                                 \
  while (j >= n) j -= n;                                                     \
                                                                             \
  if (i < m) g_odata[i] = BINARYOP(OP, g_idata[i], g_ivec[j]);               \
}

#define ELEMENT_V_KERNEL_IMPL(NAME, TYPE)                                     \
template void                                                                \
NAME##_element_v_kernel<TYPE>(TYPE* a, TYPE* v, TYPE* r, unsigned int m, unsigned int n);

#define ARRAY_ELEMENTWISE_VEC(NAME, TYPE)                                        \
  cudaError_t cudaml_ev##NAME(TYPE* a, size_t m, TYPE* v, size_t n, TYPE* r) {   \
                                                                                 \
 int threadsPerBlock = 256;                                                      \
 int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;                \
                                                                                 \
 NAME##_element_v_kernel<TYPE><<<blocksPerGrid, threadsPerBlock>>>(a, v, r, m, n); \
 return cudaGetLastError();                                                      \
}

// TODO unitary kernel
