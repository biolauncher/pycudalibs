#
# Copyright (C) 2009 Model Sciences Ltd.
#
# This file is  part of pycudalibs
#
#    pycudalibs is free software: you can redistribute it and/or modify
#    it under the terms of the Lesser GNU General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    Pycudalibs is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the Lesser GNU General Public
#    License along with pycudalibs.  If not, see <http://www.gnu.org/licenses/>.  


# python distutils setup for CUDA extension modules

from distutils.core import setup, Extension
import distutils.sysconfig as dsys
import os
import sys

# CUDA
cuda = os.getenv('CUDA_HOME')
if not cuda:
    print 'Please set CUDA_HOME to the root of the CUDA package (usually /usr/local/cuda)'
    sys.exit(1)

cuda_include = cuda + '/include'
cuda_lib = cuda + '/lib'
cuda_lib64 = cuda + '/lib64'

# CULA
cula = os.getenv('CULA_HOME')
if not cula:
    print 'Please set CULA_HOME to the root of the CULA package (usually /usr/local/cula)'
    sys.exit(2)

cula_include = cula + '/include'
cula_lib = cula + '/lib'
cula_lib64 = cula + '/lib64'


# get numpy includes
numpy_includes = None
try:
    import numpy.distutils.misc_util as ndu
    numpy_includes = ndu.get_numpy_include_dirs()
except:
    print 'Cannot import numpy.distutils - is numpy installed?'
    sys.exit(3)
if not numpy_includes:
    print 'No numpy include directories found - is numpy installed?'
    sys.exit(4)

    
# OK to try and build
includes = ['.', cula_include, cuda_include] + numpy_includes

# ensure CULA libraries come before CUDA - why?
# are 64 bit libs supported and available?
if dsys.get_config_var("SIZEOF_LONG") == 8 and os.path.exists(cuda_lib64) and os.path.exists(cula_lib64):
    library_dirs = [cula_lib64, cuda_lib64]
    print 'Building with 64 bit libraries' 
else:
    library_dirs = [cula_lib, cuda_lib]
    
#####################
# extension modules #
#####################

# libraries
BLAS = 'cublas'
LAPACK = 'cula'
#

cunumpy = Extension('_cunumpy',
                    define_macros = [
                        #('CUBLAS', '1'),
                        #('CULA', '1'),
                        ('CULA_USE_CUDA_COMPLEX', '1'),
                        ('MAJOR_VERSION', '1'),
                        ('MINOR_VERSION', '0'),
                        ('DEBUG', '1')],
                    include_dirs = includes,
                    libraries = [BLAS, LAPACK],
                    library_dirs = library_dirs,
                    sources = ['pycunumpy.c', 'pycuarray.c'])

cublas = Extension('_cublas',
                   define_macros = [
                       ('CUBLAS', '1'),
                       ('CULA', '1'),
                       ('CULA_USE_CUDA_COMPLEX', '1'),
                       ('MAJOR_VERSION', '1'),
                       ('MINOR_VERSION', '0'),
                       ('DEBUG', '0')],
                   include_dirs = includes,
                   libraries = [BLAS, LAPACK],
                   library_dirs = library_dirs,
                   sources = ['pycublas.c'])

culax = Extension('_cula',
                  define_macros = [
                      ('CUBLAS', '1'),
                      ('CULA', '1'),
                      ('CULA_USE_CUDA_COMPLEX', '1'),
                      ('MAJOR_VERSION', '1'),
                      ('MINOR_VERSION', '0'),
                      ('DEBUG', '0')],
                  include_dirs = includes,
                  libraries = [BLAS, LAPACK],
                  library_dirs = library_dirs,
                  sources = ['pycula.c'])


setup (name = 'cunumpy',
       version = '1.0',
       description = 'CUDA BLAS and CULA LAPACK integration with numpy arrays',
       author = 'Simon E. Beaumont',
       author_email = 'ix@modelsciences.com',
       url = 'http://www.modelsciences.com',
       long_description = 'APIs for CUDA and CULA libraries with support for numpy arrays. see README.',
       ext_modules = [cunumpy, cublas, culax],
       py_modules = ['cunumpy'],
       requires=['numpy(>=1.2)'])
