# make a library of kernels
# this makes a shared library:
#

UNAME		:= $(shell uname)

ifeq ($(UNAME), Darwin)
LIBEXT 		= dylib
else
LIBEXT		= so
endif

VERSION		= 0.0.2

DEBUG 		= 0
#NVCCOPTS	= --ptxas-options=-v
NVCCOPTS	= 

CUDA 		= /usr/local/cuda
CUDAINC 	= $(CUDA)/include
CUDALIB32 	= $(CUDA)/lib

# CUDA on OSX has universal libraries now
ifeq ($(UNAME), Darwin)
CUDALIB64	= $(CUDA)/lib
else
CUDALIB64 	= $(CUDA)/lib64
endif


LIBNAME 	= cudaml
INCLUDEDIR 	= ../include
ARCH32 		= -m32
ARCH64 		= -m64
LIBDIR32 	= ../lib
LIBDIR64 	= ../lib64
LIBDIRUNI	= ../universal

LIB32 		= $(LIBDIR32)/lib$(LIBNAME).$(VERSION).$(LIBEXT)
LIB64 		= $(LIBDIR64)/lib$(LIBNAME).$(VERSION).$(LIBEXT)
LIBUNI		= $(LIBDIRUNI)/lib$(LIBNAME).$(VERSION).$(LIBEXT)

KERNELS 	= null.cu reduction.cu element.cu
HEADERS 	= $(INCLUDEDIR)/cudaml.h cudautl.h reduction_kernel.h element_kernel.h

all:	lib32 lib64

lib32:	$(LIB32)

lib64:	$(LIB64)

$(LIB32):	Makefile $(KERNELS) $(HEADERS)
	mkdir -p $(LIBDIR32)
	nvcc $(ARCH32) $(NVCCOPTS) -Xcompiler -fno-strict-aliasing,-fPIC,-DDEBUG=$(DEBUG) -o $(LIB32) --shared -I$(INCLUDEDIR) -I$(CUDAINC) -I. $(KERNELS) 

$(LIB64):	Makefile $(KERNELS) $(HEADERS)
	mkdir -p $(LIBDIR32)
	nvcc $(ARCH64) $(NVCCOPTS) -Xcompiler -fno-strict-aliasing,-fPIC,-DDEBUG=$(DEBUG) -o $(LIB64) --shared -I$(INCLUDEDIR) -I$(CUDAINC) -I. $(KERNELS) 

test32:	$(LIB32) test.c
	gcc -g -DDEBUG=$(DEBUG) $(ARCH32) -o test test.c -I$(CUDAINC) -I$(INCLUDEDIR) -I. -L$(CUDALIB32) -lcuda -lcudart -L$(LIBDIR32) -l$(LIBNAME)

test64:	$(LIB64) test.c
	gcc -g -DDEBUG=$(DEBUG) $(ARCH64) -o test test.c -I$(CUDAINC) -I$(INCLUDEDIR) -I. -L$(CUDALIB64) -lcuda -lcudart -L$(LIBDIR64) -l$(LIBNAME)

universal:	lib32 lib64
	mkdir -p $(LIBDIRUNI)
	lipo -create $(LIB32) $(LIB64) -output $(LIBUNI)
clean:
	rm -f $(LIB32) $(LIB64) $(LIBUNI)
