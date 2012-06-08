pycudalibs - simple but flexible interfaces to CUDA BLAS and CULA LAPACK libraries.  
----------------------------------------------------------------------------------

We are releasing this code under LGPL3 - see copyright.txt, COPYING, COPYING.LESSER

This branch integrates the CULA LAPACK libraries and leverage the
performance improvements to CULA BLAS implementations.  CULA is
proprietary (though very reasonbly priced for research) and there is
free version with single precision functions. See:
http://www.culatools.com/

The philopsophy of this implementation is to provide cuda matrices
(and vectors) along the lines of numpy that use device functions and
kernels to implement their methods for general linear algebra, data
analysis and machine learning tasks. It should be possible to create
useful algorithms by composing array functions without incurring host
to device and device host memory transfer costs for intermediate
results or "in flight" data.

For more general utility linking the underlying BLAS/LAPACK
implementation to the CULA link libraries (which are adaptive to a
tunable job size) is more transparent and can also fail over to a high
performance CPU based implementation e.g. Intel's MKL (math kernel
library) which will leverage multi-core speedups.

More sophisticated Python/CUDA programmers wanting to enjoy the
whole CUDA, Open/CL experience will also benefit from the full monty
offered by the excellent PyCUDA package.

In short you probably don't need this unless you know otherwise.

* Synopsis: 

    import cunumpy 
    import gpu
 
* Known to work on these platforms:

    OS X 10.5.*            CUDA 2.0, 2.1 CULA R12-R14       Python 2.5-6
    OS X 10.6.*            CUDA 4.0, 4.2 CULA R12-R14       Python 2.7
    UBUNTU 11.04 (x86_64)  CUDA 4.0, 4.2 CULA R12-R14       EPD 7.1-2 (Python 2.7)

* Extras
There are some useful element and column vector kernels included which
form the basis of a machine learning package CUDAML this is built to
create a shared library libcudaml.{so,dylib} which can be linked in. I
think it safe to assume that the future of this package is CULA and
CUDAML based. We hope to leverage multi-device GPU linear algebra in
CULA R14 onwards.

________
Copyright (C) 2009-2012 Simon Beaumont - Model Sciences Ltd.

