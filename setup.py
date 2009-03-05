# Copyright (c) 2009 Simon Beaumont - All Rights Reserved

# python distutils setup for CUDA extension modules

from distutils.core import setup, Extension

# todo figure how to set these portably!
numpyinclude =  '/usr/local/src/sage-3.1.1-f/local/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/numpy/core/include'
cudainclude = '/usr/local/cuda/include'
cudalib = '/usr/local/cuda/lib'

cublas = Extension('_cublas',
                   define_macros = [('MAJOR_VERSION', '0'),
                                    ('MINOR_VERSION', '1'),
                                    ('DEBUG', '1')],
                   include_dirs = ['.', cudainclude],
                   libraries = ['cublas'],
                   library_dirs = [cudalib],
                   sources = ['pycublas.c'])

cuda = Extension('_cuda',
                 define_macros = [('MAJOR_VERSION', '0'),
                                  ('MINOR_VERSION', '1'),
                                  ('DEBUG', '1')],
                 include_dirs = ['.', cudainclude],
                 libraries = ['cublas'],
                 library_dirs = [cudalib],
                 sources = ['pycuda.c'])


setup (name = 'CUDA Libraries',
       version = '0.1',
       description = 'Low level CUDA Library APIs',
       author = 'Simon E. Beaumont',
       author_email = 'seb@modelsciences.com',
       url = 'http://www.modelsciences.com',
       long_description = '''
APIs for CUDA libraries with support for numpy arrays. Unreleased! Caveat Emptor!
''',
       ext_modules = [cuda, cublas])
