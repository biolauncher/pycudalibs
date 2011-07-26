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

# force user to set these
cuda = os.getenv('CUDA_HOME')
if not cuda:
    print 'Please set CUDA_HOME to the root of the CUDA (or CULA) package (usually /usr/local/cu(d|l)a)'
    sys.exit(1)

cuda_include = cuda + '/include'
cuda_lib = cuda + '/lib'

# actually only CULA provides this - so check for architecture and existence below
cuda_lib64 = cuda + '/lib64'


# get numpy includes
numpy_includes = None
try:
    import numpy.distutils.misc_util as ndu
    numpy_includes = ndu.get_numpy_include_dirs()
except:
    print 'Cannot import numpy.distutils - is numpy installed?'
    sys.exit(3)
if not numpy_includes:
    print 'No numpy include directories! Build may fail...'

#
includes = ['.', cuda_include] + numpy_includes

# 64bit libs supported and available?
if dsys.get_config_var("SIZEOF_LONG") == 8 and os.path.exists(cuda_lib64):
    library_dirs = [cuda_lib64]
else:
    library_dirs = [cuda_lib]
    
#####################
# extension modules #
#####################

## TODO need macros for the includes since CULA call their's culablas and NVidia cublas!
## actually these are not plug compatible at all as functions have been renamed as well as
## include files and libraries... bah!

BLAS = 'cublas'
LAPACK = 'cula'
#

cunumpy = Extension('_cunumpy',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '1'),
                                     ('DEBUG', '0')],
                    include_dirs = includes,
                    libraries = [BLAS],
                    library_dirs = library_dirs,
                    sources = ['pycunumpy.c'])

cublas = Extension('_cublas',
                   define_macros = [('MAJOR_VERSION', '0'),
                                    ('MINOR_VERSION', '1'),
                                    ('DEBUG', '0')],
                   include_dirs = includes,
                   libraries = [BLAS],
                   library_dirs = library_dirs,
                   sources = ['pycublas.c'])

# XXX todo cula extension...


setup (name = 'CuNumpy',
       version = '0.2',
       description = 'CUDA BLAS and CULA LAPACK integration with numpy arrays',
       author = 'Simon E. Beaumont',
       author_email = 'ix@modelsciences.com',
       url = 'http://www.modelsciences.com',
       long_description = '''APIs for CUDA and CULA libraries with support for numpy arrays. see README.''',
       ext_modules = [cunumpy, cublas],
       py_modules = ['cunumpy'],
       requires=['numpy(>=1.2)'])
