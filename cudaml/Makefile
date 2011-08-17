# make a library of kernels
# this makes a shared library:
# nvcc  -Xcompiler -fno-strict-aliasing,-fPIC -o libmy.dylib --shared -I/usr/local/cuda/include null.cu 
#
# than a c program can link to...
# gcc -m32 -o main main.c -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcuda -lcudart -L. -lmy

# time for autotools?
# so what is this library called: cudaml - CUDA machine learning
DEBUG=0
LIB=cudaml
THELIB = lib$(LIB).dylib
KERNELS = null.cu simple.cu centraliser.cu reduction.cu

$(THELIB):	Makefile $(KERNELS) cudautl.h cudaml.h
	nvcc  -Xcompiler -fno-strict-aliasing,-fPIC,-DDEBUG=$(DEBUG) -o $(THELIB) --shared -I/usr/local/cuda/include -I. $(KERNELS) 

test:	$(THELIB) test.c
	gcc -DDEBUG=$(DEBUG) -m32 -o test test.c -I/usr/local/cuda/include -I. -L/usr/local/cuda/lib -lcuda -lcudart -L. -l$(LIB)

