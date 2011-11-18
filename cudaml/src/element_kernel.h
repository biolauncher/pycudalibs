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
 int threadsPerBlock = THREADS_PER_BLOCK;                                                  \
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
 int threadsPerBlock = THREADS_PER_BLOCK;                                                    \
 int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;                \
                                                                                 \
 NAME##_element_v_kernel<TYPE><<<blocksPerGrid, threadsPerBlock>>>(a, v, r, m, n); \
 return cudaGetLastError();                                                      \
}

/* yet more general or is it? */

#define ELEMENT_A_KERNEL(NAME, OP)                                            \
template <class T>                                                           \
__global__ void                                                              \
 NAME##_element_a_kernel(T *g_idata, T* g_ivec, T* g_odata, unsigned int m, unsigned int n, unsigned int k) { \
  int i = blockDim.x * blockIdx.x + threadIdx.x;                             \
  int j = i / m;                                                             \
  while (j >= k) j -= k;                                                     \
                                                                             \
  if (i < m*n) g_odata[i] = BINARYOP(OP, g_idata[i], g_ivec[j]);             \
}

#define ELEMENT_A_KERNEL_IMPL(NAME, TYPE)                                                        \
template void                                                                                    \
 NAME##_element_a_kernel<TYPE>(TYPE* a, TYPE* v, TYPE* r, unsigned int m, unsigned int n, unsigned int k);

#define ARRAY_ELEMENTWISE_ARY(NAME, TYPE)                                                        \
  cudaError_t cudaml_ea##NAME(TYPE* a, size_t m, size_t n, TYPE* v, size_t k, TYPE* r) {         \
                                                                                                 \
 int threadsPerBlock = THREADS_PER_BLOCK;                                                        \
 int blocksPerGrid = (m*n + threadsPerBlock - 1) / threadsPerBlock;                              \
                                                                                                 \
 NAME##_element_a_kernel<TYPE><<<blocksPerGrid, threadsPerBlock>>>(a, v, r, m, n, k);            \
 return cudaGetLastError();                                                                      \
}

/* unary kernel */

#define ELEMENT_U_KERNEL(NAME, OP)                                            \
template <class T>                                                            \
__global__ void                                                              \
 NAME##_element_u_kernel(T *g_idata, T* g_odata, unsigned int m) {           \
  int i = blockDim.x * blockIdx.x + threadIdx.x;                             \
                                                                             \
  if (i < m) g_odata[i] = UNARYOP(OP, g_idata[i]);                           \
}

#define ELEMENT_U_KERNEL_IMPL(NAME, TYPE)                                                        \
template void                                                                                    \
NAME##_element_u_kernel<TYPE>(TYPE* a, TYPE* r, unsigned int m);

#define ARRAY_ELEMENTWISE_UNARY(NAME, TYPE)                                                      \
cudaError_t cudaml_u##NAME(TYPE* a, size_t m, TYPE* r) {                                         \
                                                                                                 \
 int threadsPerBlock = THREADS_PER_BLOCK;                                                        \
 int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;                                \
                                                                                                 \
 NAME##_element_u_kernel<TYPE><<<blocksPerGrid, threadsPerBlock>>>(a, r, m);                     \
 return cudaGetLastError();                                                                      \
}


