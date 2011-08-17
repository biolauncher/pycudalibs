/* utilities probably of more general applicability but mainly
   intended for block and thread calculations */

#ifndef __CUDAUTL_H__
#define __CUDAUTL_H__ 1

static inline int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

static inline bool isPow2(unsigned int x) {
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

#endif // __CUDAUTL_H__
