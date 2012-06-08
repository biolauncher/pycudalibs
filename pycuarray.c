// -*- mode: C; compile-command: "CULA_HOME=/usr/local/cula CUDA_HOME=/usr/local/cuda python setup.py build"; -*-
 
/*
Copyright (C) 2009-2011 Model Sciences Ltd.

This file is part of pycudalibs

    pycudalibs is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    Pycudalibs is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the Lesser GNU General Public
    License along with pycudalibs.  If not, see <http://www.gnu.org/licenses/>.  

Author
    Simon Beaumont
*/


/*******************************************
 * cuda_Array Python object implementation 
 *******************************************/

#define CUNUMPY_MODULE
#include <pycuarray.h>

/**
 * cuda_Array destructor
 */

static void
cuda_Array_dealloc(cuda_Array* self) {

  trace("TRACE cuda_Array_dealloc: %0x (%0x)\n", 
        (int) self, (int) (self->d_mem != NULL ? self->d_mem->d_ptr : 0));
  Py_XDECREF(self->a_dtype);
  Py_XDECREF(self->d_mem);
  self->ob_type->tp_free((PyObject*)self);
}


/**
 * cuda_Array __new__(cls, ...)
 */

static PyObject *
cuda_Array_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  cuda_Array *self;

  self = (cuda_Array *)type->tp_alloc(type, 0);
    
  if (self != NULL) {
    self->d_mem = NULL;
    self->e_size = 0;
    self->a_ndims = 0;
    self->a_dims[0] = 0;
    self->a_dims[1] = 0;
    self->a_dtype = NULL;
    self->a_transposed = 0; 
  }
  
  return (PyObject *)self;
}


/**
 * cuda_Array __init__(self, numpy_initializer, dtype=None)
 *  takes a numpy array or numpy style initializer
 *  and optional dtype in numpy format
 */

static int
cuda_Array_init(cuda_Array *self, PyObject *args, PyObject *kwds) {
  
  PyObject *object = NULL;
  PyArrayObject *array = NULL;
  PyArray_Descr *dtype = NULL;
  static char *kwlist [] = {"object", "dtype", NULL};

  // in case we are re-initing in __init__ 
  if (self->d_mem != NULL) { Py_DECREF(self->d_mem); }
  if (self->a_dtype != NULL) { Py_DECREF(self->a_dtype); }

  if (PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist, &object, 
                                  PyArray_DescrConverter, &dtype)) {

    // check dtype is valid
    if (dtype == NULL || !PyArray_DescrCheck(dtype))
      dtype = PyArray_DescrFromType(NPY_FLOAT32);
    
    Py_INCREF(dtype);

    // TODO: check acceptable data types
    trace("TRACE cuda_Array_init: dtype: elsize: %d domain: %s\n", 
          dtype->elsize, PyTypeNum_ISCOMPLEX(dtype->type_num) ? "complex" : "real");

    Py_INCREF(object);
    // morph supplied initialiser to a numpy array in required format, checking dimensions and dtype
    array = (PyArrayObject*) PyArray_FromAny(object, dtype, 
                                             DEVICE_ARRAY_MINDIMS, DEVICE_ARRAY_MAXDIMS, 
                                             NPY_FORTRAN | NPY_ALIGNED, NULL);
    Py_DECREF(object);

    if (array == NULL) {
      Py_DECREF(dtype);
      return -1;

    } else {

      // get and check array vital statistics for allocation of cuda device memory
      npy_intp *dims = PyArray_DIMS(array);
      int ndims = PyArray_NDIM(array);
 
      // init 
      self->a_ndims = ndims;
      self->a_dims[0] = dims[0];

      // make sure vectors are conventionally n*1 (column vectors) self->a_dims[1] = dims[1];
      self->a_dims[1] = ndims == 1 ? 1 : dims[1];
      self->e_size = dtype->elsize;

      int n_elements = a_elements(self);

      // alloc XXX now rows, cols, esize XXX
      trace("TRACE cuda_Array_init: elements: %d element_size: %d\n", n_elements, self->e_size);

      if ((self->d_mem = alloc_cuda_Memory(self->a_dims[0], self->a_dims[1], self->e_size)) == NULL) {
        Py_DECREF(array);
        Py_DECREF(dtype);
        return -1;
      }

      //Py_INCREF(self->d_mem); we own this and are not passing it to python

      // copy data from initialiser to device memory
      void* source = PyArray_DATA(array);
      
      if (cuda_error(cublasSetVector(n_elements, self->e_size, source, 1, self->d_mem->d_ptr, 1), 
                     "init:cublasSetVector")) {
        Py_DECREF(array);
        Py_DECREF(dtype);
        Py_DECREF(self->d_mem);
        return -1;
      }

      // finally update dtype in self
      self->a_dtype = dtype;
      // decref the initialiser array since we are done with it? - this fixes another leak but...
      Py_DECREF(array);
      Py_INCREF(self->a_dtype); // this is an unscientific attempt to be rid of some reference count erros
      return 0;
    }
  } else return -1;
}


/* pretty print */

static inline PyObject *
_stringify(cuda_Array *self) {
  if (self->a_ndims == 2)
    return PyString_FromFormat("<%s %p %s%d matrix(%d,%d) %d #%lu @%p>",
                               self->ob_type->tp_name,
                               self->d_mem->d_ptr,
                               PyTypeNum_ISCOMPLEX(self->a_dtype->type_num) ? "complex" : "float",
                               self->e_size * 8,
                               self->a_dims[0],
                               self->a_dims[1],
                               self->d_mem->d_pitch,
                               self->ob_refcnt, self);
  else
    return PyString_FromFormat("<%s %p %s%d vector(%d) %d #%lu @%p>",
                               self->ob_type->tp_name,
                               self->d_mem->d_ptr,
                               PyTypeNum_ISCOMPLEX(self->a_dtype->type_num) ? "complex" : "float",
                               self->e_size * 8,
                               self->a_dims[0],
                               self->d_mem->d_pitch,
                               self->ob_refcnt, self);
}

static PyObject*
cuda_Array_repr(cuda_Array *self) {
  return _stringify(self);
}

static PyObject*
cuda_Array_str(cuda_Array *self) {
  return _stringify(self);
}

/* accessor methods */

static PyObject*
cuda_Array_getShape(cuda_Array *self, void *closure) {
  if (self->a_ndims == 2) return Py_BuildValue("(ii)", self->a_dims[0], self->a_dims[1]);
  else return Py_BuildValue("(i)", self->a_dims[0]);
}

static PyObject*
cuda_Array_getDtype(cuda_Array *self, void *closure) {
  return Py_BuildValue("O", self->a_dtype);
}

static PyObject*
cuda_Array_getTranspose(cuda_Array *self, void *closure) {
  return cuda_Array_transpose(self, NULL);
}

#ifdef CULA

static PyObject*
cuda_Array_getConjugateTranspose(cuda_Array *self, void *closure) {
  return cuda_Array_conjugateTranspose(self);
}

#endif //CULA


/* create a numpy host array from cuda device array */

static PyObject*
cuda_Array_2numpyArray(cuda_Array *self, PyObject *args) {

  npy_intp dims[2];
  dims[0] = self->a_dims[0];
  dims[1] = self->a_dims[1];

  // create a numpy array to hold the data
  PyObject *array = PyArray_Empty(self->a_ndims, dims, self->a_dtype, self->a_transposed? 0 : 1);
  if (array != NULL) {
    // fill it in
    if (cuda_error(cublasGetVector (a_elements(self), self->e_size, 
                                    self->d_mem->d_ptr, 1, PyArray_DATA(array), 1),
                   "2numpy:cublasGetVector")) {
      
      Py_DECREF(array);
      return NULL;
      
    } else return Py_BuildValue("N", array);
  }
  return NULL; 
}


/***********************************
 * expose basic informational slots 
 ***********************************/

static PyMemberDef cuda_Array_members[] = {
  {"itemsize", T_INT, offsetof(cuda_Array, e_size), READONLY,
   "Size of each device array element"},
  {"ndim", T_INT, offsetof(cuda_Array, a_ndims), READONLY,
   "Number of array dimensions"},
  {NULL}
};


/***************
 * method table
 **************/

static PyMethodDef cuda_Array_methods[] = {

  {"toarray", (PyCFunction) cuda_Array_2numpyArray, METH_NOARGS,
   "Store CUDA device array into a host numpy array."},
  {"transpose", (PyCFunction) cuda_Array_transpose, METH_NOARGS,
   "Transpose of array."},
  {"dot", (PyCFunction) cuda_Array_dot, METH_VARARGS,
   "Inner product of vectors and matrices."},
  {"multiply", (PyCFunction) cuda_Array_scale, METH_VARARGS,
   "Element by element multiply."},
  {"copy", (PyCFunction) cuda_Array_copy, METH_NOARGS,
   "Create a copy of a CUDA array using only device-device transfer."},
  {"norm", (PyCFunction) cuda_Array_2norm, METH_NOARGS,
   "The 2norm of a vector or Frobenius or Hilbert-Schmidt norm of a matrix."},
  {"asum", (PyCFunction) cuda_Array_asum, METH_NOARGS,
   "The absolute sum of a CUDA device array."},
  {"reshape", (PyCFunction) cuda_Array_reshape, METH_VARARGS,
   "Reshape the dimensions of a CUDA device array."},

#ifdef CULA

  {"svd", (PyCFunction) cuda_Array_svd, METH_VARARGS | METH_KEYWORDS,
   "Singular value decomposion of CUDA array - returns tuple of (U S VT) device arrays."},
  {"eigensystem", (PyCFunction) cuda_Array_eigensystem, METH_VARARGS | METH_KEYWORDS,
   "Compute eigenvalues and optionally left and/or right eigenvectors of a square matrix."},
  {"adjoint", (PyCFunction) cuda_Array_conjugateTranspose, METH_NOARGS,
   "Conjugate transpose of a matrix"},

#endif

#if CULA >= 14

  {"pdot", (PyCFunction) cula_Array_pdot, METH_VARARGS | METH_STATIC,
   "Multi-gpu inner product of vectors and matrices N.B. arguments must be numpy (host) arrays"},

#endif // CULA>=14

#ifdef CUDAML
 
  {"sum", (PyCFunction) cuda_Array_sum, METH_NOARGS,
   "Sum of matrix/vector."},
  {"max", (PyCFunction) cuda_Array_max, METH_NOARGS,
   "Maximum value of matrix/vector."},
  {"min", (PyCFunction) cuda_Array_min, METH_NOARGS,
   "Minimum value of matrix/vector."},
  {"product", (PyCFunction) cuda_Array_product, METH_NOARGS,
   "Product of matrix/vector."},

  {"csum", (PyCFunction) cuda_Array_csum, METH_NOARGS,
   "Column sum of matrix/vector."},
  {"cmax", (PyCFunction) cuda_Array_cmax, METH_NOARGS,
   "Column max of matrix/vector."},
  {"cmin", (PyCFunction) cuda_Array_cmin, METH_NOARGS,
   "Column min of matrix/vector."},
  {"cproduct", (PyCFunction) cuda_Array_cproduct, METH_NOARGS,
   "Column product of matrix/vector."},

  {"add", (PyCFunction) cuda_Array_esum, METH_VARARGS,
   "Element by element add matrix/vector."},
  {"mul", (PyCFunction) cuda_Array_emul, METH_VARARGS,
   "Element by element multiply matrix/vector."},
  {"pow", (PyCFunction) cuda_Array_epow, METH_VARARGS,
   "Element by element matrix/vector power."},

  UNARY_FUNCTION_METHOD_TABLE_ENTRY(sqrt),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(log),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(log2),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(log10),
  
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(sin),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(cos),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(tan),
  
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(sinh),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(cosh),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(tanh),
  
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(exp),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(exp10),
  
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(sinpi),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(cospi),
  
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(asin),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(acos),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(atan),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(asinh),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(acosh),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(atanh),

  UNARY_FUNCTION_METHOD_TABLE_ENTRY(erf),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(erfc),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(erfinv),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(erfcinv),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(lgamma),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(tgamma),

  UNARY_FUNCTION_METHOD_TABLE_ENTRY(trunc),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(round),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(rint),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(floor),
  UNARY_FUNCTION_METHOD_TABLE_ENTRY(ceil),
#endif

  {NULL, NULL, 0, NULL} 
};

/**********************
 * getters and setters
 *********************/

static PyGetSetDef cuda_Array_properties[] = {
  {"shape", (getter) cuda_Array_getShape, (setter) NULL, 
   "shape of device array", NULL},
  {"dtype", (getter) cuda_Array_getDtype, (setter) NULL, 
   "dtype of device array", NULL},
  {"T", (getter) cuda_Array_getTranspose, (setter) NULL, 
   "transpose of array", NULL},
#ifdef CULA
  {"H", (getter) cuda_Array_getConjugateTranspose, (setter) NULL,
   "conjugate transpose of array", NULL},
#endif // CULA
  {NULL}
};


/**************
 * object type
 **************/

static PyTypeObject cuda_ArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /*ob_size*/
    CUDA_ARRAY_TYPE_NAME,                     /*tp_name*/
    sizeof(cuda_Array),                       /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)cuda_Array_dealloc,           /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    (reprfunc)cuda_Array_repr,                /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    (reprfunc)cuda_Array_str,                 /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "CUDA device array",                      /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    cuda_Array_methods,                       /* tp_methods */
    cuda_Array_members,                       /* tp_members */
    cuda_Array_properties,                    /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)cuda_Array_init,                /* tp_init */
    0,                                        /* tp_alloc */
    cuda_Array_new,                           /* tp_new */
};


/**
 * virtual transpose  
 */

static PyObject*
cuda_Array_transpose(cuda_Array *self, PyObject *args) {
  // toggle the transpose flag for matrices only since we keep vectors in column form
  // XXX this may lose on pitch optimization
  if (!isvector(self)) {

    cuda_Array *clone = (cuda_Array *) _PyObject_New(&cuda_ArrayType);
    if (clone != NULL) {

      clone->e_size = self->e_size;
      clone->a_ndims = self->a_ndims;
      clone->a_transposed = (self->a_transposed ^ 1);
      clone->a_dims[0] = self->a_dims[1];
      clone->a_dims[1] = self->a_dims[0];

      clone->d_mem = self->d_mem; 
      Py_INCREF(clone->d_mem);
      clone->a_dtype = self->a_dtype;
      Py_INCREF(clone->a_dtype);
      return Py_BuildValue("N", clone); // this is a new reference

    } else return NULL;
  
  } else return Py_BuildValue("O", self); // noop for vectors as they are always column vecs 
}



/**
 * dot product - emulates numpy array and vector behaviour for taking inner-products
 * the mother of all methods - and this is just the single precision version!
 * we allocate and return result matrices as cuda_Array where appropriate but none of
 * this inccurs device <-> host memory transfers.
 * this method is overloaded for vectors and matrices with 64bit complex and float32 payloads
 */

static PyObject*
cuda_Array_dot(cuda_Array *self, PyObject *args) {
  cuda_Array *other;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (PyArg_ParseTuple(args, "O!", &cuda_ArrayType, &other)) {
    
    // types match?
    if (!PyArray_EquivTypes(self->a_dtype, other->a_dtype)) {
      PyErr_SetString(PyExc_ValueError, "array types are not equivalent");
      return NULL;
    }
    
    /* leading dimensions for matrices are swapped as dims have already been! 
       n.b. vectors are never transposed */

    int lda = self->a_transposed ? self->a_dims[1] : self->a_dims[0];
    int ldb = other->a_transposed ? other->a_dims[1] : other->a_dims[0];
    
    if (isvector(self)) {

      if (isvector(other)) {
        // vector-vector sdot or cdot -> scalar float
        if (lda != ldb) {
          PyErr_SetString(PyExc_ValueError, "vectors have different dimensions");
          return NULL;
        }

        if (iscomplex(self)) {
          cuComplex cip = cublasCdotu(lda, self->d_mem->d_ptr, 1, other->d_mem->d_ptr, 1);
          if (cublas_error("cdotu")) return NULL;
          else {
            Py_complex ip;
            ip.real = (double) cip.x;
            ip.imag = (double) cip.y;
            return Py_BuildValue("D", &ip);
          }

        } else {
            float ip = cublasSdot(lda, self->d_mem->d_ptr, 1, other->d_mem->d_ptr, 1);
            if (cublas_error("sdot")) return NULL;
            else return Py_BuildValue("f", ip);
        }
       
      } else {
        // vector-matrix -> pretend vector is a 1,n matrix (row vector) and do c/sgemm
        // but return a conventional vector
        int m = 1;
        int k = self->a_dims[0];
        int kb = other->a_dims[0];
        int n = other->a_dims[1];

        if (k != kb) {
          PyErr_SetString(PyExc_ValueError, "arrays have wrong shapes for vector-matrix inner product");
          return NULL;
        }
        // create a conventional vector to receive result
        cuda_Array *C = make_vector(n, self->a_dtype);
        if (C == NULL) return NULL;
        int ldc = 1;

        if (iscomplex(self)) {
          cuComplex alpha = {1., 0.};
          cuComplex beta = {0., 0.};

          cublasCgemm(OP(self), OP(other), m, n, k, alpha, 
                      self->d_mem->d_ptr, lda, other->d_mem->d_ptr, ldb, beta, C->d_mem->d_ptr, ldc);
          if (cublas_error("cgemm")) return NULL;
        
        } else {
          cublasSgemm(OP(self), OP(other), m, n, k, 1., 
                      self->d_mem->d_ptr, lda, other->d_mem->d_ptr, ldb, 0., C->d_mem->d_ptr, ldc);
          if (cublas_error("sgemm")) return NULL;
        }

        return Py_BuildValue("N", C);
      }
      
    } else {
      
      if (isvector(other)) {
        // matrix-vector -> conventional (column) vector
        int m = self->a_dims[0];
        int k = self->a_dims[1];
        int kb = other->a_dims[0];
        int n = 1;

        if (k != kb) {
          PyErr_SetString(PyExc_ValueError, "arrays have wrong shapes for matrix-vector inner product");
          return NULL;
        }
        // create a conventional vector to receive result
        cuda_Array *C = make_vector(m, self->a_dtype);
        if (C == NULL) return NULL;
        int ldc = m;

        if (iscomplex(self)) {
          cuComplex alpha = {1., 0.};
          cuComplex beta = {0., 0.};

          cublasCgemm(OP(self), OP(other), m, n, k, alpha, 
                      self->d_mem->d_ptr, lda, other->d_mem->d_ptr, ldb, beta, C->d_mem->d_ptr, ldc);
          if (cublas_error("cgemm")) return NULL;
        
        } else {
          cublasSgemm(OP(self), OP(other), m, n, k, 1., 
                      self->d_mem->d_ptr, lda, other->d_mem->d_ptr, ldb, 0., C->d_mem->d_ptr, ldc);
          if (cublas_error("sgemm")) return NULL;
        }

        return Py_BuildValue("N", C);
      
      } else {
        // matrix-matrix sgemm or cgemm
        
        int m = self->a_dims[0];
        int k = self->a_dims[1];
        int kb = other->a_dims[0];
        int n = other->a_dims[1];

        if (k != kb) {
          PyErr_SetString(PyExc_ValueError, "arrays have wrong shapes for matrix-matrix inner product");
          return NULL;
        }

        // create a matrix
        cuda_Array *C = make_matrix(m, n, self->a_dtype);
        if (C == NULL) return NULL;
        int ldc = m;

        if (iscomplex(self)) {
          cuComplex alpha = {1., 0.};
          cuComplex beta = {0., 0.};

          cublasCgemm(OP(self), OP(other), m, n, k, alpha, 
                      self->d_mem->d_ptr, lda, other->d_mem->d_ptr, ldb, beta, C->d_mem->d_ptr, ldc);
          if (cublas_error("cgemm")) return NULL;
        
        } else {
          cublasSgemm(OP(self), OP(other), m, n, k, 1., 
                      self->d_mem->d_ptr, lda, other->d_mem->d_ptr, ldb, 0., C->d_mem->d_ptr, ldc);
          if (cublas_error("sgemm")) return NULL;
        }

        return Py_BuildValue("N", C);
      }
    } 
  }
  else return NULL;
}


/**
 * scalar multiplication 
 */

static PyObject*
cuda_Array_scale(cuda_Array *self, PyObject *args) {
  PyObject *scalar;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (PyArg_ParseTuple(args, "O", &scalar)) {

    // no nasty side effects here - copy is cheap (well device side anyway)
    cuda_Array *copy = copy_array(self);

    if (PyNumber_Check(scalar)) {
      
      if (PyComplex_Check(scalar)) {

        if (iscomplex(copy)) {

          Py_complex z_scalar = PyComplex_AsCComplex(scalar);
          cuComplex c_scalar;
          c_scalar.x = (float) z_scalar.real;
          c_scalar.y = (float) z_scalar.imag;
          
          // call blas complex-complex case
          cublasCscal(a_elements(copy), c_scalar, copy->d_mem->d_ptr, 1);
          return cublas_error("cscal") ? NULL : Py_BuildValue("N", copy);
            
        } else {
          PyErr_SetString(PyExc_ValueError, "cannot scale real array with complex scalar - yet!");
          return NULL;
        }

      } else {
        // real scalar
        float s = (float) PyFloat_AsDouble(scalar);

        if (iscomplex(copy)) {
          
          cublasCsscal(a_elements(copy), s, copy->d_mem->d_ptr, 1);
          return cublas_error("csscal") ? NULL : Py_BuildValue("N", copy);
       
        } else {
          
          cublasSscal(a_elements(copy), s, copy->d_mem->d_ptr, 1);
          return cublas_error("sscal") ? NULL : Py_BuildValue("N", copy);
        }
      }

    } else {
      PyErr_SetString(PyExc_ValueError, "can only mutiply by scalars - watch this space!");
      return NULL;
    }
    
  } else return NULL;
}


/**
 * 2 norm euclidean L2 for vectors or Frobenius or Hilbert-Schmidt for matrices.
 * trickery is to treat matrices as vectors to achieve entrywise norms 
 */

static PyObject*
cuda_Array_2norm(cuda_Array *self) {

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (iscomplex(self)) {
    float norm = cublasScnrm2(a_elements(self), self->d_mem->d_ptr, 1);
    return cublas_error("scnrm2") ? NULL : Py_BuildValue("f", norm);

  } else {
    float norm = cublasSnrm2(a_elements(self), self->d_mem->d_ptr, 1);
    return cublas_error("snrm2") ? NULL : Py_BuildValue("f", norm);
  }
}


/**
 * absolute sum of an array - not the L1 norm in complex case
 */  

static PyObject*
cuda_Array_asum(cuda_Array *self) {

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (iscomplex(self)) {
    float sum = cublasScasum(a_elements(self), self->d_mem->d_ptr, 1);
    return cublas_error("scasum") ? NULL : Py_BuildValue("f", sum);

  } else {
    float sum = cublasSasum(a_elements(self), self->d_mem->d_ptr, 1);
    return cublas_error("sasum") ? NULL : Py_BuildValue("f", sum);
  }
}


/**
 * reshape - clone and meddle with the descriptor
 */

static PyObject*
cuda_Array_reshape(cuda_Array *self, PyObject *args) {
  //PyObject* tuple;
  int dim0 = 0, dim1 = 0;

  if (PyArg_ParseTuple(args, "(ii)", &dim0, &dim1)) {
  
    int ndims = dim1 == 0 ? 1 : 2;
    if (ndims == 1) dim1 = 1; // column vector convention

    if (dim0 == 0 || dim0 * dim1 != self->a_dims[0] * self->a_dims[1]) {
      PyErr_SetString(PyExc_ValueError, "required reshape does not match existing array dimensions");
      return NULL;
    }
      
    cuda_Array *clone = (cuda_Array *) _PyObject_New(&cuda_ArrayType);
    if (clone != NULL) {

      clone->e_size = self->e_size;
      clone->a_ndims = ndims; // RESHAPE
      clone->a_transposed = self->a_transposed;

      clone->a_dims[0] = dim0; // RESHAPE
      clone->a_dims[1] = dim1; // RESHAPE

      clone->d_mem = self->d_mem; 
      Py_INCREF(clone->d_mem);
      clone->a_dtype = self->a_dtype;
      Py_INCREF(clone->a_dtype);
      return Py_BuildValue("N", clone);

    } else return NULL;
  } else return NULL;
}


/**
 * a method to create a copy of an array in cuda space
 */

static PyObject* 
cuda_Array_copy(cuda_Array *self) {
  cuda_Array *copy = copy_array(self);
  return (copy == NULL) ? NULL : Py_BuildValue("N", copy);
}


/*************************
 * LAPACK - CULA methods 
 *************************/
#ifdef CULA

static PyObject*
cuda_Array_svd(cuda_Array* self, PyObject* args, PyObject* keywds) {

  // dimensions
  int m = self->a_dims[0];
  int n = self->a_dims[1];
  int lda = max(1,m);
  int ldu = m;
  int ldvt = min(m,n);

  static char* kwlist[] = {"pure", NULL};
  int pure = 1;

  /* parse Python aguments */
  if (PyArg_ParseTupleAndKeywords(args, keywds, "|i", kwlist, &pure)) {

    // allocate device arrays to recieve svd results
    cuda_Array *U = make_matrix(m, min(m,n), self->a_dtype);
    if (U == NULL) return NULL;
  
    // S is always real
    cuda_Array *S = make_vector(min(m,n), dtype(isdouble(self) ? NPY_FLOAT64 : NPY_FLOAT32));
    if (S == NULL) return NULL;
  
    cuda_Array *VT = make_matrix(min(m,n), min(m,n), self->a_dtype);
    if (VT == NULL) return NULL;

    /* copy device memory */
    void* A = pure ? copy_devmem(self) : self->d_mem->d_ptr;
    if (A == NULL) return NULL;

    /* LAPACK dispatch based on array type */
    void (*LAPACK_fn)();
 
    if (iscomplex(self))
      LAPACK_fn = isdouble(self) ? (void (*)()) culaDeviceZgesvd : (void (*)()) culaDeviceCgesvd;
    else
      LAPACK_fn = isdouble(self) ? (void (*)()) culaDeviceDgesvd : (void (*)()) culaDeviceSgesvd;
  

    /* N.B. array A contents are always destroyed by LAPACK! */
    
    (*LAPACK_fn)('S', 'S', self->a_dims[0], self->a_dims[1], 
                 A, lda,
                 S->d_mem->d_ptr,
                 U->d_mem->d_ptr, ldu,
                 VT->d_mem->d_ptr, ldvt);
  
    if (culablas_error("device_gesvd")) {
      if (pure && cula_error(culaDeviceFree(A), "dealloc:culaDeviceFreeA"));
      return NULL;

    } else {
      /* free copied device memory */
      if (pure && cula_error(culaDeviceFree(A), "dealloc:culaDeviceFreeA")) return NULL;
      else return Py_BuildValue("OOO", U, S, VT);
    }

  } else return NULL; // argument error
}


/** 
 *  eigensystem  - values, and/or left right vectors 
 *    most of the work here is dealing with the awful LAPACK api.
 */

static PyObject*
cuda_Array_eigensystem(cuda_Array* self, PyObject* args, PyObject* keywds) {

  /* default to eigenvalues only */
  int leftv = 0;
  int rightv = 0;
  int pure = 1; // by default we copy our argument so as not to have side-effects

  static char* kwlist[] = {"pure", "left_vectors", "right_vectors", NULL};

  /* check matrix is square */
  if (self->a_dims[0] != self->a_dims[1]) {
    PyErr_SetString(PyExc_ValueError, "matrix must be square for eigensolver");
    return NULL;
  }
  
  /* parse Python aguments */
  if (PyArg_ParseTupleAndKeywords(args, keywds, "|iii", kwlist, &pure, &leftv, &rightv)) {

    /* 
       to maintain purity we just copy the device mem and explicitly dealloc 
       temporary cuda memory before exit.
       (we also apply this approach to the dummy eigenvectors below)
    */
    void* A = pure ? copy_devmem(self) : self->d_mem->d_ptr;
    if (A == NULL) return NULL;

    /* GEEV params */
    char jobvl = leftv ? 'V' : 'N';
    char jobvr = rightv ? 'V' : 'N';
    int n = self->a_dims[0];
    int lda = max(1,n);

    /*
      NOTA - left and right eigenvectors:

      For real variants (S/D): If the j-th eigenvalue is real, then
      u(j) = VL(:,j), the j-th column of VL. If the j-th and (j+1)-st
      eigenvalues form a complex conjugate pair, then u(j) = VL(:,j) +
      i*VL(:,j+1) and u(j+1) = VL(:,j) - i*VL(:,j+1).
      
      For complex variants (C/Z): u(j) = VL(:,j), the j-th column of VL.
    */

    /* XXX contrary to the claims of the documentation we cannot pass
       a null pointer (or even a host pointer) when 'N' is supplied as
       they are dereferenced in GEEV - so we fake some temps. YETCH! */

    /* left eigenvectors */
    int ldvl = leftv ? n : 1;
    cuda_Array* vl = leftv ? make_matrix(ldvl, ldvl, self->a_dtype) : NULL;
    if (leftv && vl == NULL) goto die;

    /* create temp dummy device mem if not returning left vector */
    void* VL = leftv ? vl->d_mem->d_ptr : deviceAllocTmp(1, 1, self->e_size);
    if (VL == NULL) goto die;

    /* right eigenvectors */
    int ldvr = rightv ? n : 1;
    cuda_Array* vr = rightv ? make_matrix(ldvr, ldvr, self->a_dtype) : NULL;
    if (rightv && vr == NULL) goto die;

    /* create tmp dummy device mem if not returning right vector */
    void* VR = rightv ? vr->d_mem->d_ptr : deviceAllocTmp(1, 1, self->e_size);
    if (VR == NULL) goto die;

    /* LAPACK dispatch based on array type */
    void (*LAPACK_fn)();
 
    if (iscomplex(self)) {

      cuda_Array* w = make_vector(n, self->a_dtype);
      if (w == NULL) goto die;

      LAPACK_fn = isdouble(self) ? (void (*)()) culaDeviceZgeev : (void (*)()) culaDeviceCgeev;
      (*LAPACK_fn)(jobvl, jobvr, n, 
                   A, lda, 
                   w->d_mem->d_ptr, 
                   VL, ldvl, 
                   VR, ldvr);
      
      /* free any temporary CUDA memory */
      if (pure && cula_error(culaDeviceFree(A), "dealloc:culaDeviceFreeA")) goto die;
      if (!leftv && cula_error(culaDeviceFree(VL), "dealloc:culaDeviceFreeVL")) goto die;
      if (!rightv && cula_error(culaDeviceFree(VR), "dealloc:culaDeviceFreeVR")) goto die;

      if (culablas_error("deviceC|Zgesvd"))
        goto die;
      else if (leftv && rightv)
        return Py_BuildValue("NNN", vl, w, vr);
      else if (leftv)
        return Py_BuildValue("NN", vl, w);
      else if (rightv)
        return Py_BuildValue("NN", w, vr);
      else 
        return Py_BuildValue("N", w);

    } else {

      cuda_Array* wr = make_vector(n, self->a_dtype);
      if (wr == NULL) goto die;
      cuda_Array* wi = make_vector(n, self->a_dtype);
      if (wi == NULL) goto die;

      LAPACK_fn = isdouble(self) ? (void (*)()) culaDeviceDgeev : (void (*)()) culaDeviceSgeev;
      (*LAPACK_fn)(jobvl, jobvr, n, 
                   A, lda, 
                   wr->d_mem->d_ptr, wi->d_mem->d_ptr, 
                   VL, ldvl, 
                   VR, ldvr);

      /* free any temporary CUDA memory */
      if (pure && cula_error(culaDeviceFree(A), "dealloc:culaDeviceFreeA")) goto die;
      if (!leftv && cula_error(culaDeviceFree(VL), "dealloc:culaDeviceFreeVL")) goto die;
      if (!rightv && cula_error(culaDeviceFree(VR), "dealloc:culaDeviceFreeVR")) goto die;

      /* check on LAPACK call */
      if (culablas_error("deviceS|Dgesvd"))
        goto die;
    
      else if (leftv && rightv)
        return Py_BuildValue("OOOO", vl, wr, wi, vr);
      else if (leftv)
        return Py_BuildValue("OOO", vl, wr, wi);
      else if (rightv)
        return Py_BuildValue("OOO", wr, wi, vr);
      else 
        return Py_BuildValue("OO", wr, wi);
    }

  die:
    /* cleanup any temporary allocations and exit*/
    if (pure && A != NULL)
      (void) cula_error(culaDeviceFree(A), "die-dealloc:culaDeviceFreeA");
    if (!leftv && VL != NULL)
      (void) cula_error(culaDeviceFree(VL), "die-dealloc:culaDeviceFreeVL");
    if (!rightv && VR != NULL)
      (void) cula_error(culaDeviceFree(VR), "die-dealloc:culaDeviceFreeVR");
    
    return NULL;
    
  } else return NULL;
}


/* 
** conjugate transpose 
*/

static PyObject*
cuda_Array_conjugateTranspose(cuda_Array* self) {

  if (!iscomplex(self)) {
    PyErr_SetString(PyExc_ValueError, "can only take conjugate transpose of complex array");
    return NULL;
  }
  
  int m = self->a_dims[0];
  int n = self->a_dims[1];
  
  cuda_Array* c = isvector(self) ? make_vector(m, self->a_dtype) : make_matrix(n, m, self->a_dtype);
  if (c == NULL) return NULL;

  void (*LAPACK_fn)() = isdouble(self) ? 
    (void (*)()) culaDeviceZgeTransposeConjugate : (void (*)()) culaDeviceCgeTransposeConjugate;

  (*LAPACK_fn)(m, n, self->d_mem->d_ptr, max(1,m), c->d_mem->d_ptr, max(1,n));
  if (culablas_error("deviceGeTransposeConjugate")) 
    return NULL;
  else 
    return Py_BuildValue("N", c);
}
#endif


#if CULA >= 14
/*
** experimental (alpha) CULA multi-gpu dot product
** 
** N.B. multi-gpu array methods need host memory for arrays so this is a static method in the class
** that takes two regular numpy arrays or suitable array initialisers (so this is really a numpy extension)
*/

static PyObject*
cula_Array_pdot(PyObject *null, PyObject *args) {

  PyObject* arg1;
  PyObject* arg2;

  PyArrayObject* array1;
  PyArrayObject* array2;
  PyArray_Descr* dtype;

  pculaConfig pcula;
  if (cula_error(pculaConfigInit(&pcula), "cula_Array_dot:pCulaConfigInit")) return NULL;


  if (PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {

    dtype = PyArray_DescrFromType(NPY_FLOAT32);


    // convert args to suitable numpy arrays
    Py_INCREF(dtype);
    Py_INCREF(arg1);
    array1 = (PyArrayObject*) PyArray_FromAny(arg1, dtype, 
                                             DEVICE_ARRAY_MINDIMS, DEVICE_ARRAY_MAXDIMS, 
                                             NPY_FORTRAN | NPY_ALIGNED, NULL);
    Py_DECREF(arg1);
    if (array1 == NULL) {
      Py_DECREF(dtype);
      return NULL;
    }

    Py_INCREF(dtype);
    Py_INCREF(arg2);
    array2 = (PyArrayObject*) PyArray_FromAny(arg2, dtype, 
                                             DEVICE_ARRAY_MINDIMS, DEVICE_ARRAY_MAXDIMS, 
                                             NPY_FORTRAN | NPY_ALIGNED, NULL);
    Py_DECREF(arg2);

    if (array2 == NULL) {
      Py_DECREF(dtype);
      return NULL;
    }

    // 1. need to check array dimensions etc.
    npy_intp* dims1 = PyArray_DIMS(array1);
    int ndims1 = PyArray_NDIM(array1);

    npy_intp* dims2 = PyArray_DIMS(array2);
    int ndims2 = PyArray_NDIM(array2);
    
    int lda = dims1[0];
    int ldb = dims2[0];
    int m, n, k, ldc;
    char opA = 'N', opB = 'N';

    npy_intp dimsR[2];
    int ndimsR;

    if (ndims1 == 1) {

      if (ndims2 == 1) {

        // vector-vector 
        if (dims1[0] != dims2[0]) {
          PyErr_SetString(PyExc_ValueError, "vectors have different dimensions");
          Py_DECREF(array1);
          Py_DECREF(array2);
          Py_DECREF(dtype);
          return NULL;
        }

        // pretend vector1 is a 1,k matrix or row vector and vector2 is a k,1 matrix or column vector
        opA = 'T';
        m = 1;
        k = dims1[0];
        n = 1;
        ldc = m;

        // return a element 0-d array 
        dimsR[0] = 0;
        ndimsR = 0;

      } else {
        // vector-matrix - pretend vector1 is a 1,k matrix or row vector
        opA = 'T';
        m = 1;
        k = dims1[0];
          
        if (k != dims2[0]) {
          PyErr_SetString(PyExc_ValueError, "arrays have wrong shape for vector-matrix inner product");
          Py_DECREF(array1);
          Py_DECREF(array2);
          Py_DECREF(dtype);
          return NULL;
        }
        n = dims2[1];
        ldc = m;
        // return a 1-d array
        dimsR[0] = n;
        ndimsR = 1;
      }

    } else if (ndims2 == 1) {
      // matrix-vector vector2 is a k,1 column vector
      m = dims1[0];
      k = dims1[1];

      if (k != dims2[0]) {
        PyErr_SetString(PyExc_ValueError, "arrays have wrong shape for matrix-vector inner product");
        Py_DECREF(array1);
        Py_DECREF(array2);
        Py_DECREF(dtype);
        return NULL;
      }
      n = dims2[1];
      ldc = m;
      // return a 1-d array
      dimsR[0] = m;
      ndimsR = 1;

    } else {
      // matrix-matrix
      m = dims1[0];
      k = dims1[1];

      if (k != dims2[0]) {
        PyErr_SetString(PyExc_ValueError, "arrays have wrong shape for matrix-matrix inner product");
        Py_DECREF(array1);
        Py_DECREF(array2);
        Py_DECREF(dtype);
        return NULL;
      }
      n = dims2[1];
      ldc = m;
      // return a 2-d array 
      dimsR[0] = m;
      dimsR[1] = n;
      ndimsR = 2;
    }

      
    Py_INCREF(dtype);
    // 2. create a numpy (fortran) array to hold the result
    PyObject* result = PyArray_Zeros(ndimsR, dimsR, dtype, 1);
    if (result == NULL) {
      Py_DECREF(array1);
      Py_DECREF(array2);
      Py_DECREF(dtype);
      return NULL;
    }

    // 3. get pointers to the raw data
    culaFloat* A = (culaFloat*) PyArray_DATA(array1);
    culaFloat* B = (culaFloat*) PyArray_DATA(array2);
    culaFloat* C = (culaFloat*) PyArray_DATA((PyArrayObject*) result);

    // 4. call the lapack routine
    // printf("m=%d n=%d k=%d lda=%d ldb=%d ldc=%d\n", m, n, k, lda, ldb, ldc);
    if (cula_error(pculaSgemm(&pcula, opA, opB, m, n, k, (culaFloat) 1., A, lda, B, ldb, (culaFloat) 0., C, ldc),
                   "pdot:pculaSgemm")) {

      Py_DECREF(array1);
      Py_DECREF(array2);
      Py_DECREF(dtype);
      Py_DECREF(result);
      return NULL;
    }

    // 5. clean up 
    Py_DECREF(array1);
    Py_DECREF(array2);
    Py_DECREF(dtype);

    // 5. return the array
    return PyArray_Return((PyArrayObject*) result);

  } else return NULL;
}
#endif

#ifdef CUDAML
/*********************************************************************************
 * CULAML custom ml kernel library - mainly element-wise and column-wise functions
 *********************************************************************************/

static PyObject*
cuda_Array_sum(cuda_Array* self) {

  float sum;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision reduction not yet implemented");
    return NULL;
  }

  if (cuda_error2(cudaml_asum(self->d_mem->d_ptr, a_elements(self), &sum), "cuda_Array_sum"))
    return NULL;
  else
    return Py_BuildValue("f", sum);
}


static PyObject*
cuda_Array_max(cuda_Array* self) {

  float sum;
  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision reduction not yet implemented");
    return NULL;
  }

  if (cuda_error2(cudaml_amax(self->d_mem->d_ptr, a_elements(self), &sum), "cuda_Array_max"))
    return NULL;
  else
    return Py_BuildValue("f", sum);
}


static PyObject*
cuda_Array_min(cuda_Array* self) {

  float sum;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision reduction not yet implemented");
    return NULL;
  }

  if (cuda_error2(cudaml_amin(self->d_mem->d_ptr, a_elements(self), &sum), "cuda_Array_min"))
    return NULL;
  else
    return Py_BuildValue("f", sum);
}


static PyObject*
cuda_Array_product(cuda_Array* self) {

  float sum;
  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision reduction not yet implemented");
    return NULL;
  }

  if (cuda_error2(cudaml_aproduct(self->d_mem->d_ptr, a_elements(self), &sum), "cuda_Array_product"))
    return NULL;
  else
    return Py_BuildValue("f", sum);
}


static PyObject*
cuda_Array_csum(cuda_Array* self) {

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision reduction not yet implemented");
    return NULL;
  }

  int m = self->a_dims[0];
  int n = self->a_dims[1];

  // allocate a new vector for the result
  cuda_Array* colv = make_vector(n, self->a_dtype);

  if (cuda_error2(cudaml_csum(self->d_mem->d_ptr, m, n, colv->d_mem->d_ptr), "cuda_Array_csum"))
    return NULL;
  else
    return Py_BuildValue("N", colv);
}


static PyObject*
cuda_Array_cmax(cuda_Array* self) {

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision reduction not yet implemented");
    return NULL;
  }

  int m = self->a_dims[0];
  int n = self->a_dims[1];

  // allocate a new vector for the result
  cuda_Array* colv = make_vector(n, self->a_dtype);

  if (cuda_error2(cudaml_cmax(self->d_mem->d_ptr, m, n, colv->d_mem->d_ptr), "cuda_Array_cmax"))
    return NULL;
  else
    return Py_BuildValue("N", colv);
}


static PyObject*
cuda_Array_cmin(cuda_Array* self) {

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision reduction not yet implemented");
    return NULL;
  }

  int m = self->a_dims[0];
  int n = self->a_dims[1];

  // allocate a new vector for the result
  cuda_Array* colv = make_vector(n, self->a_dtype);

  if (cuda_error2(cudaml_cmin(self->d_mem->d_ptr, m, n, colv->d_mem->d_ptr), "cuda_Array_cmin"))
    return NULL;
  else
    return Py_BuildValue("N", colv);
}


static PyObject*
cuda_Array_cproduct(cuda_Array* self) {

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision reduction not yet implemented");
    return NULL;
  }

  int m = self->a_dims[0];
  int n = self->a_dims[1];

  // allocate a new vector for the result
  cuda_Array* colv = make_vector(n, self->a_dtype);

  if (cuda_error2(cudaml_cproduct(self->d_mem->d_ptr, m, n, colv->d_mem->d_ptr), "cuda_Array_cproduct"))
    return NULL;
  else
    return Py_BuildValue("N", colv);
}


static PyObject*
cuda_Array_esum(cuda_Array *self, PyObject *args) {
  PyObject *arg;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (PyArg_ParseTuple(args, "O", &arg)) {

    // no nasty side effects here - copy is cheap (device side anyway)
    cuda_Array *copy = copy_array(self);

    if (PyNumber_Check(arg)) {
      
      if (PyComplex_Check(arg)) {

        // dont do complex element wise yet
        PyErr_SetString(PyExc_ValueError, "complex scalars not yet implemented");
        return NULL;
        
        /*
        if (iscomplex(copy)) {

          Py_complex z_scalar = PyComplex_AsCComplex(scalar);
          cuComplex c_scalar;
          c_scalar.x = (float) z_scalar.real;
          c_scalar.y = (float) z_scalar.imag;
          
        }
        */

      } else {
        // real scalar
        float s = (float) PyFloat_AsDouble(arg);

        if (iscomplex(copy)) {
          PyErr_SetString(PyExc_ValueError, "complex arrays not yet implemented");
          return NULL;
          
        } else {

          return cuda_error2(cudaml_esum(self->d_mem->d_ptr, a_elements(self), s, copy->d_mem->d_ptr), 
                             "cuda_Array_esum") ? NULL : Py_BuildValue("N", copy);
        }
      }

    } else {

      // argument could be an another array...
      cuda_Array* other;

      if (PyArg_ParseTuple(args, "O!", &cuda_ArrayType, &other)) {
    
        // types match?
        if (!PyArray_EquivTypes(self->a_dtype, other->a_dtype)) {
          PyErr_SetString(PyExc_ValueError, "array types are not equivalent");
          return NULL;
        }

        // check dimensions and use relevant kernel
        /* if number of columns matches vector then broadcast to columns */
        if (ismatrix(self) && self->a_dims[1] == other->a_dims[0]) {

          return cuda_error2(cudaml_easum(self->d_mem->d_ptr, self->a_dims[0], self->a_dims[1], 
                                          other->d_mem->d_ptr, other->a_dims[0],
                                          copy->d_mem->d_ptr), 
                             "cuda_Array_easum") ? NULL : Py_BuildValue("N", copy);
        
   
        } else { 
          /* broadcast the vector to the array */
          return cuda_error2(cudaml_evsum(self->d_mem->d_ptr, a_elements(self), 
                                          other->d_mem->d_ptr, a_elements(other),
                                          copy->d_mem->d_ptr), 
                             "cuda_Array_evsum") ? NULL : Py_BuildValue("N", copy);
        }

      } else {
        PyErr_SetString(PyExc_ValueError, "argument type not applicable to this array");
        return NULL;
      }
    }
    
  } else return NULL;
}


static PyObject*
cuda_Array_emul(cuda_Array *self, PyObject *args) {
  PyObject *arg;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (PyArg_ParseTuple(args, "O", &arg)) {

    // no nasty side effects here - copy is cheap (device side anyway)
    cuda_Array *copy = copy_array(self);

    if (PyNumber_Check(arg)) {
      
      if (PyComplex_Check(arg)) {

        // dont do complex element wise yet
        PyErr_SetString(PyExc_ValueError, "complex scalars not yet implemented");
        return NULL;
        
        /*
        if (iscomplex(copy)) {

          Py_complex z_scalar = PyComplex_AsCComplex(scalar);
          cuComplex c_scalar;
          c_scalar.x = (float) z_scalar.real;
          c_scalar.y = (float) z_scalar.imag;
          
        }
        */

      } else {
        // real scalar
        float s = (float) PyFloat_AsDouble(arg);

        if (iscomplex(copy)) {
          PyErr_SetString(PyExc_ValueError, "complex arrays not yet implemented");
          return NULL;
          
        } else {

          return cuda_error2(cudaml_emul(self->d_mem->d_ptr, a_elements(self), s, copy->d_mem->d_ptr), 
                             "cuda_Array_emul") ? NULL : Py_BuildValue("N", copy);
        }
      }

    } else {

      // argument could be an another array...
      cuda_Array* other;

      if (PyArg_ParseTuple(args, "O!", &cuda_ArrayType, &other)) {
    
        // types match?
        if (!PyArray_EquivTypes(self->a_dtype, other->a_dtype)) {
          PyErr_SetString(PyExc_ValueError, "array types are not equivalent");
          return NULL;
        }

        if (ismatrix(self) && self->a_dims[1] == other->a_dims[0]) {
          /* broadcast to columns */
          return cuda_error2(cudaml_eamul(self->d_mem->d_ptr, self->a_dims[0], self->a_dims[1], 
                                          other->d_mem->d_ptr, other->a_dims[0],
                                          copy->d_mem->d_ptr), 
                             "cuda_Array_eamul") ? NULL : Py_BuildValue("N", copy);
        
   
        } else { 
          /* broadcast the vector to the array */
          return cuda_error2(cudaml_evmul(self->d_mem->d_ptr, a_elements(self), 
                                          other->d_mem->d_ptr, a_elements(other),
                                          copy->d_mem->d_ptr), 
                             "cuda_Array_evmul") ? NULL : Py_BuildValue("N", copy);
        }
   
      } else {
        PyErr_SetString(PyExc_ValueError, "argument type not applicable to this array");
        return NULL;
      }
    }
    
  } else return NULL;
}


static PyObject*
cuda_Array_epow(cuda_Array *self, PyObject *args) {
  PyObject *arg;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (PyArg_ParseTuple(args, "O", &arg)) {

    // no nasty side effects here - copy is cheap (device side anyway)
    cuda_Array *copy = copy_array(self);

    if (PyNumber_Check(arg)) {
      
      if (PyComplex_Check(arg)) {

        // dont do complex element wise yet
        PyErr_SetString(PyExc_ValueError, "complex scalars not yet implemented");
        return NULL;
        
        /*
        if (iscomplex(copy)) {

          Py_complex z_scalar = PyComplex_AsCComplex(scalar);
          cuComplex c_scalar;
          c_scalar.x = (float) z_scalar.real;
          c_scalar.y = (float) z_scalar.imag;
          
        }
        */

      } else {
        // real scalar
        float s = (float) PyFloat_AsDouble(arg);

        if (iscomplex(copy)) {
          PyErr_SetString(PyExc_ValueError, "complex arrays not yet implemented");
          return NULL;
          
        } else {

          return cuda_error2(cudaml_epow(self->d_mem->d_ptr, a_elements(self), s, copy->d_mem->d_ptr), 
                             "cuda_Array_epow") ? NULL : Py_BuildValue("N", copy);
        }
      }

    } else {

      // argument could be an another array...
      cuda_Array* other;

      if (PyArg_ParseTuple(args, "O!", &cuda_ArrayType, &other)) {
    
        // types match?
        if (!PyArray_EquivTypes(self->a_dtype, other->a_dtype)) {
          PyErr_SetString(PyExc_ValueError, "array types are not equivalent");
          return NULL;
        }

        if (ismatrix(self) && self->a_dims[1] == other->a_dims[0]) {
          /* broadcast to columns */
          return cuda_error2(cudaml_eapow(self->d_mem->d_ptr, self->a_dims[0], self->a_dims[1], 
                                          other->d_mem->d_ptr, other->a_dims[0],
                                          copy->d_mem->d_ptr), 
                             "cuda_Array_eapow") ? NULL : Py_BuildValue("N", copy);
        
   
        } else { 
          /* broadcast the vector to the array */
          return cuda_error2(cudaml_evpow(self->d_mem->d_ptr, a_elements(self), 
                                          other->d_mem->d_ptr, a_elements(other),
                                          copy->d_mem->d_ptr), 
                             "cuda_Array_evpow") ? NULL : Py_BuildValue("N", copy);
        }
   
      } else {
        PyErr_SetString(PyExc_ValueError, "argument type not applicable to this array");
        return NULL;
      }
    }
    
  } else return NULL;
}


/* unary math methods */
#define UNARY_FUNCTION_METHOD(FUNC)                                                                    \
static PyObject*                                                                                       \
cuda_Array_##FUNC(cuda_Array *self) {                                                                  \
                                                                                                       \
  if (isdouble(self)) {                                                                                \
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented"); \
    return NULL;                                                                                       \
  }                                                                                                    \
                                                                                                       \
  cuda_Array *copy = copy_array(self);                                                                 \
                                                                                                       \
  return cuda_error2(cudaml_u##FUNC(self->d_mem->d_ptr, a_elements(self), copy->d_mem->d_ptr),         \
                     "cuda_Array_u" #FUNC) ? NULL : Py_BuildValue("N", copy);                          \
}

UNARY_FUNCTION_METHOD(sqrt)
UNARY_FUNCTION_METHOD(log)
UNARY_FUNCTION_METHOD(log2)
UNARY_FUNCTION_METHOD(log10)

UNARY_FUNCTION_METHOD(sin)
UNARY_FUNCTION_METHOD(cos)
UNARY_FUNCTION_METHOD(tan)

UNARY_FUNCTION_METHOD(sinh)
UNARY_FUNCTION_METHOD(cosh)
UNARY_FUNCTION_METHOD(tanh)

UNARY_FUNCTION_METHOD(exp)
UNARY_FUNCTION_METHOD(exp10)

UNARY_FUNCTION_METHOD(sinpi)
UNARY_FUNCTION_METHOD(cospi)

UNARY_FUNCTION_METHOD(asin)
UNARY_FUNCTION_METHOD(acos)
UNARY_FUNCTION_METHOD(atan)
UNARY_FUNCTION_METHOD(asinh)
UNARY_FUNCTION_METHOD(acosh)
UNARY_FUNCTION_METHOD(atanh)

UNARY_FUNCTION_METHOD(erf)
UNARY_FUNCTION_METHOD(erfc)
UNARY_FUNCTION_METHOD(erfinv)
UNARY_FUNCTION_METHOD(erfcinv)
UNARY_FUNCTION_METHOD(lgamma)
UNARY_FUNCTION_METHOD(tgamma)

UNARY_FUNCTION_METHOD(trunc)
UNARY_FUNCTION_METHOD(round)
UNARY_FUNCTION_METHOD(rint)
UNARY_FUNCTION_METHOD(floor)
UNARY_FUNCTION_METHOD(ceil)

#endif // CUDAML

/***************************************************************************
 * helper methods for constructing arrays and vectors these are not part of
 * the public api
 ***************************************************************************/

static inline cuda_Array*
make_vector(int n, PyArray_Descr *dtype) {

  cuda_Array *self = (cuda_Array *) _PyObject_New(&cuda_ArrayType);
  //trace("make_vector: (%d)\n", n);

  if (self != NULL) {
    self->e_size = dtype->elsize;
    self->a_ndims = 1;
    self->a_dims[0] = n;
    self->a_dims[1] = 1; // column vector convention
    self->a_transposed = 0; 
    self->d_mem = NULL;

    // alloc XXX now rows, cols, esize XXX
    if ((self->d_mem = alloc_cuda_Memory(self->a_dims[0], self->a_dims[1], self->e_size)) == NULL) {
      return NULL;
    }

    self->a_dtype = dtype;
    Py_INCREF(self->a_dtype);
  }
  return self;
}


static inline cuda_Array*
make_matrix(int m, int n, PyArray_Descr *dtype) {

  cuda_Array *self = (cuda_Array *) _PyObject_New(&cuda_ArrayType);
  //trace("make_matrix: (%d,%d)\n", m, n);

  if (self != NULL) {
    self->e_size = dtype->elsize;
    self->a_ndims = 2;
    self->a_dims[0] = m;
    self->a_dims[1] = n; 
    self->a_transposed = 0; 
    self->d_mem = NULL;

    // alloc XXX now rows, cols, esize XXX
    if ((self->d_mem = alloc_cuda_Memory(m, n, self->e_size)) == NULL) {
      return NULL;
    }

    self->a_dtype = dtype;
    Py_INCREF(self->a_dtype);
  }
  return self;
}


static inline cuda_Array* 
copy_array(cuda_Array* self) {
  
  if (isdouble(self)) {
    /* could use cudaMemcopy device to device for double precision here... */
    // cudaError_t sts cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyDeviceToDevice);
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;
  }

  cuda_Array *new = (cuda_Array *) _PyObject_New(&cuda_ArrayType);

  if (new != NULL) {
    new->e_size = self->e_size;
    new->a_ndims = self->a_ndims;
    new->a_dims[0] = self->a_dims[0];
    new->a_dims[1] = self->a_dims[1]; 
    new->a_transposed = self->a_transposed; 
    new->d_mem = NULL;

    // alloc device memory
    if ((new->d_mem = alloc_cuda_Memory(new->a_dims[0], new->a_dims[1], new->e_size)) == NULL) {
      return NULL;
    }

    if (iscomplex(self)) {
      // single precsion complex
      cublasCcopy (a_elements(self), self->d_mem->d_ptr, 1, new->d_mem->d_ptr, 1);
      if (cublas_error("ccopy")) return NULL;

    } else {
      // copy self to new using blas vector copy: single precision real
      cublasScopy (a_elements(self), self->d_mem->d_ptr, 1, new->d_mem->d_ptr, 1);
      if (cublas_error("scopy")) return NULL;
    }

    new->a_dtype = self->a_dtype;
    Py_INCREF(new->a_dtype);
  }
  return new;
}


/*
 * this just copies the device memory in an array 
 *  handy for temporary buffers when LAPACK calls are side affecting and we wish
 *  to preserve purity.
 */

static inline void*
copy_devmem(cuda_Array* self) {

  void* dst = deviceAllocTmp(self->a_dims[0], self->a_dims[1], self->e_size);
  if (dst == NULL) return NULL;

  if (cuda_error2(cudaMemcpy (dst, self->d_mem->d_ptr, a_size(self), cudaMemcpyDeviceToDevice),
                  "cudaMemory:cudaMemcpy2d:copy_devmem"))
    return NULL;
                  
  else return dst;  
}


/*
 * 2d array copies with pitched mem
 */
static inline void*
copy_devmem2d(cuda_Array* self) {
  
  void* dst;
  int dpitch;

  //TODO understand the pitch bitch and hw our fortran arrays map to 2d 
  //TODO check if we are copying a matrix or a vector and use the appropriate allocator
  //TODO move all this to pycumem.h and standardise memory allocation
  // cudaError_t cudaMallocPitch (void ∗∗ dst, size_t ∗ dpitch, size_t width, size_t height)

  if (cula_error(culaDeviceMalloc((void**) &dst, &dpitch, self->a_dims[0], self->a_dims[1], self->e_size),
                 "cuda_Memory:culaDeviceMalloc:copy_devmem2d"))
    return NULL;
  
  int width = self->a_dims[1]; // columns
  int height = self->a_dims[0]; // rows

  trace("cudaMemcpy2D: dpitch=%d, spitch=%d, width=%d, height=%d\n", dpitch, self->d_mem->d_pitch, width, height);
  // 2d array copy dev2dev XXX don't understand width and height here yet!
  if (cuda_error2(cudaMemcpy2D (dst, (size_t) dpitch, 
                                self->d_mem->d_ptr, self->d_mem->d_pitch, 
                                width, 
                                height, 
                                cudaMemcpyDeviceToDevice), 
                  "cudaMemory:cudaMemcpy2d:copy_devmem2d"))
    return NULL;
  
  else return dst;
}


/*******************
 * Type descriptors
 *******************/

static inline PyArray_Descr* 
dtype(int typenum) {
  PyArray_Descr* dtype = PyArray_DescrNewFromType(typenum);
  //Py_INCREF(dtype);
  return dtype;
}


/*********************
 * Module definitions 
 *********************/

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

static PyMethodDef module_methods[] = {
  {NULL}  /* Sentinel */
};


PyMODINIT_FUNC
init_cunumpy(void) {
    PyObject* module;

    if (PyType_Ready(&cuda_ArrayType) < 0)
      return;

    if (init_cuda_MemoryType() < 0)
      return;

    module = Py_InitModule3(CUDA_MODULE_NAME, module_methods, "CUDA numpy style array module.");

    if (module == NULL) return;
    
    else {
      Py_INCREF(&cuda_ArrayType);
      PyModule_AddObject(module, CUDA_ARRAY_TYPE_SYM_NAME, (PyObject *) &cuda_ArrayType);

      cuda_exception = PyErr_NewException(CUDA_ERROR_TYPE_NAME, NULL, NULL);
      Py_INCREF(cuda_exception);
      PyModule_AddObject(module, CUDA_ERROR_TYPE_SYM_NAME, cuda_exception);

      import_array(); // import numpy module
    }
}

// END pycuarray.c //
