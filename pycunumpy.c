/* Copyright (C) 2009 Simon Beaumont - All Rights Reserved */

/**
 * numpy integration with cuda device memory, defines module: _cunumpy
 */

#define CUNUMPY_MODULE
#include <pycunumpy.h>

static void
cuda_Array_dealloc(cuda_Array* self) {

  trace("TRACE cuda_Array_dealloc: %0x (%0x)\n", (int) self, (int) (self->d_mem != NULL ? self->d_mem->d_ptr : 0));
  Py_XDECREF(self->a_dtype);
  Py_XDECREF(self->d_mem);
  self->ob_type->tp_free((PyObject*)self);
}


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
 * numpy array to cuda device memory constructor: 
 *  takes a numpy array
 *  an optional dtype in numpy format
 *  returns a cuda DeviceMemory copy of the array suitable for cublas
 *  TODO: rework this in terms of new allocators.
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

  //if (PyArg_ParseTuple(args, "O|O&", &object, PyArray_DescrConverter, &dtype)) {
  if (PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist, &object, PyArray_DescrConverter, &dtype)) {

    // default typecast
    if (dtype == NULL) dtype = PyArray_DescrFromType(NPY_FLOAT32);
    Py_INCREF(dtype);

    // TODO: check acceptable data types
    trace("TRACE cuda_Array_init: dtype: elsize: %d domain: %s\n", 
          dtype->elsize, PyTypeNum_ISCOMPLEX(dtype->type_num) ? "complex" : "real");

    Py_INCREF(object);
    // cast supplied initialiser to a numpy array in required format checking dimensions
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

      // alloc
      trace("TRACE cuda_Array_init: elements: %d element_size: %d\n", n_elements, self->e_size);

      if ((self->d_mem = alloc_cuda_Memory(n_elements, self->e_size)) == NULL) {
        Py_DECREF(array);
        Py_DECREF(dtype);
        return -1;
      }

      //Py_INCREF(self->d_mem);
      // copy data from initialiser to device memory
      void* source = PyArray_DATA(array);
      
      if (cuda_error(cublasSetVector(n_elements, self->e_size, source, 1, self->d_mem->d_ptr, 1), 
                     "init:cublasSetVector")) {
        Py_DECREF(array);
        Py_DECREF(dtype);
        Py_DECREF(self->d_mem); //???
        return -1;
      }

      // finally update dtype in self
      self->a_dtype = dtype;
      return 0;
    }
  } else return -1;
}


/* pretty print etc. */

static inline PyObject *
_stringify(cuda_Array *self) {
  if (self->a_ndims == 2)
    return PyString_FromFormat("<%s %p %s%d matrix(%d,%d) @%p>",
                               self->ob_type->tp_name,
                               self->d_mem->d_ptr,
                               PyTypeNum_ISCOMPLEX(self->a_dtype->type_num) ? "complex" : "float",
                               self->e_size * 8,
                               self->a_dims[0],
                               self->a_dims[1],
                               self);
  else
    return PyString_FromFormat("<%s %p %s%d vector(%d) @%p>",
                               self->ob_type->tp_name,
                               self->d_mem->d_ptr,
                               PyTypeNum_ISCOMPLEX(self->a_dtype->type_num) ? "complex" : "float",
                               self->e_size * 8,
                               self->a_dims[0],
                               self);
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
  {"toarray", (PyCFunction) cuda_Array_2numpyArray, METH_VARARGS,
   "Store CUDA device array into a host numpy array."},
  {"transpose", (PyCFunction) cuda_Array_transpose, METH_VARARGS,
   "Transpose of array."},
  {"dot", (PyCFunction) cuda_Array_dot, METH_VARARGS,
   "Inner product of vectors and matrices."},
  {"multiply", (PyCFunction) cuda_Array_scale, METH_VARARGS,
   "Element by element multiply."},

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
    "CUDA device backed array",               /* tp_doc */
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
      return Py_BuildValue("O", clone);

    } else return NULL;
  
  } else return Py_BuildValue("O", self); // noop for vectors as they are always column vecs 
}


/**
 * dot product - emulates numpy array and vector behaviour for taking inner-products
 * the monster of all methods - and this is just the single precision version!
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

        return Py_BuildValue("O", C);
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

        return Py_BuildValue("O", C);
      
      } else {
        // matrix-matrix sgemm or cgemm
        int m = self->a_dims[0];
        int k = self->a_dims[1];
        int kb = other->a_dims[0];
        int n = other->a_dims[1];

        if (k != kb) {
          PyErr_SetString(PyExc_ValueError, "arrays have wrong shapes for matrix-vector inner product");
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

        return Py_BuildValue("O", C);
      }
    } 
  }
  else return NULL;
}

/**
 * scalar multiplication - in place! yetch maybe we should copy the thing and then we can do 
 * scaling by complex of real vectors (arrays)
 * TODO - this is exposed as mutiply (make it do element by element also)
 */
static PyObject*
cuda_Array_scale(cuda_Array *self, PyObject *args) {
  PyObject *scalar;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (PyArg_ParseTuple(args, "O", &scalar)) {
    if (PyNumber_Check(scalar)) {
      
      if (PyComplex_Check(scalar)) {

        if (iscomplex(self)) {

          Py_complex z_scalar = PyComplex_AsCComplex(scalar);
          cuComplex c_scalar;
          c_scalar.x = (float) z_scalar.real;
          c_scalar.y = (float) z_scalar.imag;
          
          // call blas complex-complex case
          cublasCscal(a_elements(self), c_scalar, self->d_mem->d_ptr, 1);
          return cublas_error("cscal") ? NULL : Py_BuildValue("O", self);
            
        } else {
          PyErr_SetString(PyExc_ValueError, "cannot scale real array with complex scalar - yet!");
          return NULL;
        }

      } else {
        // real scalar
        float s = (float) PyFloat_AsDouble(scalar);

        if (iscomplex(self)) {
          
          cublasCsscal(a_elements(self), s, self->d_mem->d_ptr, 1);
          return cublas_error("csscal") ? NULL : Py_BuildValue("O", self);
       
        } else {
          
          cublasSscal(a_elements(self), s, self->d_mem->d_ptr, 1);
          return cublas_error("sscal") ? NULL : Py_BuildValue("O", self);
        }
      }

    } else {
      PyErr_SetString(PyExc_ValueError, "scalar is not a number");
      return NULL;
    }
    
  } else return NULL;
}

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

    //if (cuda_error(cublasAlloc(n, self->e_size, (void**)&self->d_ptr), "make_vector:cublasAlloc")) {
    if ((self->d_mem = alloc_cuda_Memory(n, self->e_size)) == NULL) {
      return NULL;
    }

    self->a_dtype =  dtype;
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

    //if (cuda_error(cublasAlloc(m*n, self->e_size, (void**)&self->d_ptr), "make_matrix:cublasAlloc")) {
    if ((self->d_mem = alloc_cuda_Memory(m*n, self->e_size)) == NULL) {
      return NULL;
    }

    self->a_dtype =  dtype;
    Py_INCREF(self->a_dtype);
  }
  return self;
}


/*******************
 * module functions
 ******************/

static PyMethodDef module_methods[] = {
  {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
init_cunumpy(void) {
    PyObject* module;

    if (PyType_Ready(&cuda_ArrayType) < 0)
      return;

    if (init_cuda_MemoryType() < 0)
      return;

    module = Py_InitModule3(CUDA_MODULE_NAME, module_methods, "CUDA device memory array module.");

    if (module == NULL) return;
    
    else {
      Py_INCREF(&cuda_ArrayType);
      PyModule_AddObject(module, CUDA_ARRAY_TYPE_SYM_NAME, (PyObject *)&cuda_ArrayType);

      cuda_exception = PyErr_NewException(CUDA_ERROR_TYPE_NAME, NULL, NULL);
      Py_INCREF(cuda_exception);
      PyModule_AddObject(module, CUDA_ERROR_TYPE_SYM_NAME, cuda_exception);

      import_array(); // import numpy module
    }
}
