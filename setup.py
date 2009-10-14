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
import os
import sys

# force user to set these
cuda = os.getenv('CUDA_HOME')
if not cuda:
    print 'Please set CUDA_HOME'
    sys.exit(1)

# is distutils numpy aware?
numpyinclude = os.getenv('NUMPY_INCLUDE')
if not numpyinclude:
    print 'Please set NUMPY_INCLUDE'
    sys.exit(3)

cudainclude = cuda + '/include'
cudalib = cuda + '/lib'

# extension modules

cunumpy = Extension('_cunumpy',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '1'),
                                     ('DEBUG', '0')],
                    include_dirs = ['.', cudainclude, numpyinclude],
                    libraries = ['cublas'],
                    library_dirs = [cudalib],
                    sources = ['pycunumpy.c'])

cublas = Extension('_cublas',
                   define_macros = [('MAJOR_VERSION', '0'),
                                    ('MINOR_VERSION', '1'),
                                    ('DEBUG', '0')],
                   include_dirs = ['.', cudainclude, numpyinclude],
                   libraries = ['cublas'],
                   library_dirs = [cudalib],
                   sources = ['pycublas.c'])


setup (name = 'CuNumpy',
       version = '0.1',
       description = 'CUDA BLAS integration with numpy arrays',
       author = 'Simon E. Beaumont',
       author_email = 'seb@modelsciences.com',
       url = 'http://www.modelsciences.com',
       long_description = '''
APIs for CUDA libraries with support for numpy arrays. see README.
''',
       ext_modules = [cunumpy, cublas],
       py_modules = ['cunumpy'],
       requires=['numpy(>=1.2)']
       )
