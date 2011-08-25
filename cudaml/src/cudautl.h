/* utilities probably of more general applicability but mainly
   intended for block and thread calculations */

#ifndef __CUDAUTL_H__
#define __CUDAUTL_H__ 1


#define F2DI(m, i, j) (((m)*(i))+(j)) 

static inline int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

static inline int isPow2(unsigned int x) {
  return ((x&(x-1))==0);
}

// compute reduction kernel threads and block sizes for matrices/vectors of size n
#define MAX_THREADS 256
#define MAX_BLOCKS 64

static inline int rk_threads(int n) {
  return (n < MAX_THREADS * 2) ? nextPow2((n + 1)/ 2) : MAX_THREADS;
}
        
static inline int rk_blocks(int n, int threads) {
  return min(MAX_BLOCKS, (n + (threads * 2 - 1)) / (threads * 2));
}

static inline int mk_threads(int n) {
  return rk_threads(n);
}

static inline int mk_blocks(int n, int threads) {
  return (n + (threads * 2 - 1)) / (threads * 2); 
}


#if DEBUG > 0
#warning "tracing calls will be compiled"
#include <stdio.h>
#define trace(format, ...) fprintf(stderr, format, ## __VA_ARGS__)
#else
#define trace(format, ...)
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#warning "printf calls from device are not supported"
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

#endif // __CUDAUTL_H__
