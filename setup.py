# Copyright (c) 2009 Simon Beaumont - All Rights Reserved

# python distutils setup for CUDA extension modules

from distutils.core import setup, Extension

# todo figure how to set these portably!
numpyinclude =  '/Applications/sage/local/lib/python2.5/site-packages/numpy/core/include/numpy'
cudainclude = '/usr/local/cuda/include'
cudalib = '/usr/local/cuda/lib'

# extension modules

cublas = Extension('_cublas',
                   define_macros = [('MAJOR_VERSION', '0'),
                                    ('MINOR_VERSION', '1'),
                                    ('DEBUG', '1')],
                   include_dirs = ['.', cudainclude],
                   libraries = ['cublas'],
                   library_dirs = [cudalib],
                   sources = ['pycublas.c'])

cudamem = Extension('_cudamem',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '1'),
                                     ('DEBUG', '1')],
                    include_dirs = ['.', cudainclude],
                    libraries = ['cublas'],
                    library_dirs = [cudalib],
                    sources = ['pycudamem.c'])

cunumpy = Extension('_cunumpy',
                    define_macros = [('MAJOR_VERSION', '0'),
                                      ('MINOR_VERSION', '1'),
                                      ('DEBUG', '1')],
                     include_dirs = ['.', cudainclude, numpyinclude],
                     libraries = ['cublas'],
                     library_dirs = [cudalib],
                     sources = ['pycunumpy.c'])


setup (name = 'CUDA Libraries',
       version = '0.1',
       description = 'Low level CUDA Library APIs',
       author = 'Simon E. Beaumont',
       author_email = 'seb@modelsciences.com',
       url = 'http://www.modelsciences.com',
       long_description = '''
APIs for CUDA libraries with support for numpy arrays. Unreleased! Caveat Emptor!
''',
       ext_modules = [cudamem, cublas, cunumpy])
