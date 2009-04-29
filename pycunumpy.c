/* Copyright (C) 2009 Simon Beaumont - All Rights Reserved */

/**
 * numpy integration with cuda device memory, defines module: _cunumpy
 */

#define CUNUMPY_MODULE
#include <pycunumpy.h>

static void
cuda_DeviceMemory_dealloc(cuda_DeviceMemory* self) {

  trace("TRACE cuda_DeviceMemory_dealloc: %0x (%0x)\n", (int) self, (int) self->d_ptr);

  Py_XDECREF(self->a_dtype);
  self->ob_type->tp_free((PyObject*)self);

  /* TODO refcoount these only sensible thing is to create another python object
     as we now share d_ptr e.g. in clones like transpose */
 
  if (self->d_ptr != NULL) 
    if (cuda_error(cublasFree(self->d_ptr), "dealloc:cublasFree"))
      return;
}


static PyObject *
cuda_DeviceMemory_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  cuda_DeviceMemory *self;

  self = (cuda_DeviceMemory *)type->tp_alloc(type, 0);
    
  if (self != NULL) {
    self->d_ptr = NULL;
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
cuda_DeviceMemory_init(cuda_DeviceMemory *self, PyObject *args, PyObject *kwds) {
  
  PyObject *object = NULL;
  PyArrayObject *array = NULL;
  PyArray_Descr *dtype = NULL;
  static char *kwlist [] = {"object", "dtype", NULL};

  // in case we are re-initing __init__ free old memory 
  if (self->d_ptr != NULL && cuda_error(cublasFree(self->d_ptr), "init:cublasFree"))
    return -1;

  if (self->a_dtype != NULL) { Py_DECREF(self->a_dtype); }

  //if (PyArg_ParseTuple(args, "O|O&", &object, PyArray_DescrConverter, &dtype)) {
  if (PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist, &object, PyArray_DescrConverter, &dtype)) {

    // default typecast
    if (dtype == NULL) dtype = PyArray_DescrFromType(NPY_FLOAT32);
    Py_INCREF(dtype);

    // TODO: check acceptable data types
    trace("TRACE cuda_DeviceMemory_init: dtype: elsize: %d domain: %s\n", 
          dtype->elsize, PyTypeNum_ISCOMPLEX(dtype->type_num) ? "complex" : "real");

    Py_INCREF(object);
    // cast supplied initialiser to a numpy array in required format checking dimensions
    array = (PyArrayObject*) PyArray_FromAny(object, dtype, 
                                             DEVICE_ARRAY_MINDIMS, DEVICE_ARRAY_MAXDIMS, 
                                             NPY_FORTRAN | NPY_ALIGNED, NULL);
    Py_DECREF(object);

    if (array == NULL) {
      return -1;
      Py_DECREF(dtype);

    } else {
      // get and check array vital statistics for allocation of cuda device memory
      npy_intp *dims = PyArray_DIMS(array);
      int ndims = PyArray_NDIM(array);

      // may lift this restriction in time - now checked in PyArray_FromAny
      //if (ndims < 1 || ndims > 2) {
      //  PyErr_SetString(PyExc_TypeError, "number of array dimensions must be 1 or 2");
      //  Py_DECREF(array);
      //  return -1;
      //}

      // init 
      self->a_ndims = ndims;
      self->a_dims[0] = dims[0];
      // make sure vectors are conventionally n*1 (column vectors) self->a_dims[1] = dims[1];
      self->a_dims[1] = ndims == 1 ? 1 : dims[1];
      self->e_size = dtype->elsize;

      int n_elements = a_elements(self);

      // alloc
      trace("TRACE cuda_DeviceMemory_init: elements: %d element_size: %d\n", n_elements, self->e_size);
    
      if (cuda_error(cublasAlloc(n_elements, self->e_size, (void**)&self->d_ptr), 
                     "init:cublasAlloc")) {
        Py_DECREF(array);
        Py_DECREF(dtype);
        return -1;
      }

      // copy data from initialiser to device memory
      void* source = PyArray_DATA(array);
      
      if (cuda_error(cublasSetVector(n_elements, self->e_size, source, 1, self->d_ptr, 1), 
                     "init:cublasSetVector")) {
        Py_DECREF(array);
        Py_DECREF(dtype);
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
_stringify(cuda_DeviceMemory *self) {
  if (self->a_ndims == 2)
    return PyString_FromFormat("<%s %p %s%d matrix(%d,%d) @%p>",
                               self->ob_type->tp_name,
                               self->d_ptr,
                               PyTypeNum_ISCOMPLEX(self->a_dtype->type_num) ? "complex" : "float",
                               self->e_size * 8,
                               self->a_dims[0],
                               self->a_dims[1],
                               self);
  else
    return PyString_FromFormat("<%s %p %s%d vector(%d) @%p>",
                               self->ob_type->tp_name,
                               self->d_ptr,
                               PyTypeNum_ISCOMPLEX(self->a_dtype->type_num) ? "complex" : "float",
                               self->e_size * 8,
                               self->a_dims[0],
                               self);
}

static PyObject*
cuda_DeviceMemory_repr(cuda_DeviceMemory *self) {
  return _stringify(self);
}

static PyObject*
cuda_DeviceMemory_str(cuda_DeviceMemory *self) {
  return _stringify(self);
}

/* accessor methods */

static PyObject*
cuda_DeviceMemory_getShape(cuda_DeviceMemory *self, void *closure) {
  if (self->a_ndims == 2) return Py_BuildValue("(ii)", self->a_dims[0], self->a_dims[1]);
  else return Py_BuildValue("(i)", self->a_dims[0]);
}

static PyObject*
cuda_DeviceMemory_getDtype(cuda_DeviceMemory *self, void *closure) {
  return Py_BuildValue("O", self->a_dtype);
}

static PyObject*
cuda_DeviceMemory_getTranspose(cuda_DeviceMemory *self, void *closure) {
  return cuda_DeviceMemory_transpose(self, NULL);
}


/* create a numpy host array from cuda device array */

static PyObject*
cuda_DeviceMemory_2numpyArray(cuda_DeviceMemory *self, PyObject *args) {

  npy_intp dims[2];
  dims[0] = self->a_dims[0];
  dims[1] = self->a_dims[1];

  // create a numpy array to hold the data
  PyObject *array = PyArray_Empty(self->a_ndims, dims, self->a_dtype, self->a_transposed? 0 : 1);
  if (array != NULL) {
    // fill it in
    if (cuda_error(cublasGetVector (a_elements(self), self->e_size, 
                                    self->d_ptr, 1, PyArray_DATA(array), 1),
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

static PyMemberDef cuda_DeviceMemory_members[] = {
  {"itemsize", T_INT, offsetof(cuda_DeviceMemory, e_size), READONLY,
   "Size of each device array element"},
  {"ndim", T_INT, offsetof(cuda_DeviceMemory, a_ndims), READONLY,
   "Number of array dimensions"},
  {NULL}
};


/***************
 * method table
 **************/

static PyMethodDef cuda_DeviceMemory_methods[] = {
  {"toarray", (PyCFunction) cuda_DeviceMemory_2numpyArray, METH_VARARGS,
   "copy cuda device array to a host numpy array."},
  {"transpose", (PyCFunction) cuda_DeviceMemory_transpose, METH_VARARGS,
   "get transpose of array."},
  {"dot", (PyCFunction) cuda_DeviceMemory_dot, METH_VARARGS,
   "inner product the final dot!"},
  {NULL, NULL, 0, NULL} 
};

/**********************
 * getters and setters
 *********************/

static PyGetSetDef cuda_DeviceMemory_properties[] = {
  {"shape", (getter) cuda_DeviceMemory_getShape, (setter) NULL, 
   "shape of device array", NULL},
  {"dtype", (getter) cuda_DeviceMemory_getDtype, (setter) NULL, 
   "dtype of device array", NULL},
  {"T", (getter) cuda_DeviceMemory_getTranspose, (setter) NULL, 
   "transpose of array", NULL},
  {NULL}
};


/**************
 * object type
 **************/

static PyTypeObject cuda_DeviceMemoryType = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /*ob_size*/
    CUDA_ARRAY_TYPE_NAME,                     /*tp_name*/
    sizeof(cuda_DeviceMemory),                /*tp_basicsize*/
    0,                                        /*tp_itemsize*/
    (destructor)cuda_DeviceMemory_dealloc,    /*tp_dealloc*/
    0,                                        /*tp_print*/
    0,                                        /*tp_getattr*/
    0,                                        /*tp_setattr*/
    0,                                        /*tp_compare*/
    (reprfunc)cuda_DeviceMemory_repr,         /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    (reprfunc)cuda_DeviceMemory_str,          /*tp_str*/
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
    cuda_DeviceMemory_methods,                /* tp_methods */
    cuda_DeviceMemory_members,                /* tp_members */
    cuda_DeviceMemory_properties,             /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)cuda_DeviceMemory_init,         /* tp_init */
    0,                                        /* tp_alloc */
    cuda_DeviceMemory_new,                    /* tp_new */
};

/**
 * virtual transpose - TODO: create a copy array with shared pointer to device memory
 * which means reference counting the device memory - needing an object wrapper of its own
 * to manage the de/allocation -  or we do our own struct with a counter! This is how
 * numpy works as the underlying data is shared! We could keep a reference to our "parent"
 * in the clone and increment that dependancy - in a chain... might work. but could
 * lead to nasty garbage chains since object are kept alive for no good reason.
 * no shortcuts to good code eh? 
 *
 */
static PyObject*
cuda_DeviceMemory_transpose(cuda_DeviceMemory *self, PyObject *args) {
  // toggle the transpose flag for matrices only since we keep vectors in column form
  if (!isvector(self)) {

    cuda_DeviceMemory *clone = (cuda_DeviceMemory *) _PyObject_New(&cuda_DeviceMemoryType);
    if (clone != NULL) {

      clone->e_size = self->e_size;
      clone->a_ndims = self->a_ndims;
      clone->a_transposed = (self->a_transposed ^ 1);
      clone->a_dims[0] = self->a_dims[1];
      clone->a_dims[1] = self->a_dims[0];

      /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         shared object: must not deallocate now until refcounted - leak or be damned 
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
      clone->d_ptr = self->d_ptr; 

      clone->a_dtype = self->a_dtype;
      Py_INCREF(clone->a_dtype);
      return Py_BuildValue("O", clone);

    } else return NULL;
  
  } else return (PyObject*) self; // noop for vectors as they are always column vecs 
}


/**
 * dot product - emulates numpy array and vector behaviour for taking inner-products
 * the monster of all methods!
 */
static PyObject*
cuda_DeviceMemory_dot(cuda_DeviceMemory *self, PyObject *args) {
  cuda_DeviceMemory *other;

  if (isdouble(self)) {
    PyErr_SetString(PyExc_NotImplementedError, "double precision linear algebra not yet implemented");
    return NULL;

  } else if (PyArg_ParseTuple(args, "O!", &cuda_DeviceMemoryType, &other)) {
    
    // types match?
    if (!PyArray_EquivTypes(self->a_dtype, other->a_dtype)) {
      PyErr_SetString(PyExc_ValueError, "array types are not equivalent");
      return NULL;
    }

    int lda = self->a_dims[0];
    int ldb = other->a_dims[0];
    
    if (isvector(self)) {

      if (isvector(other)) {
        // vector-vector sdot or cdot -> scalar float
        if (lda != ldb) {
          PyErr_SetString(PyExc_ValueError, "vectors have different dimensions");
          return NULL;
        }

        if (iscomplex(self)) {
          cuComplex cip = cublasCdotu(lda, self->d_ptr, 1, other->d_ptr, 1);
          if (cublas_error("cdotu")) return NULL;
          else {
            Py_complex ip;
            ip.real = (double) cip.x;
            ip.imag = (double) cip.y;
            return Py_BuildValue("D", &ip);
          }

        } else {
            float ip = cublasSdot(lda, self->d_ptr, 1, other->d_ptr, 1);
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
        cuda_DeviceMemory *C = make_vector(n, self->a_dtype);
        if (C == NULL) return NULL;
        int ldc = 1;

        if (iscomplex(self)) {
          cuComplex alpha = {1., 0.};
          cuComplex beta = {0., 0.};

          cublasCgemm(OP(self), OP(other), m, n, k, alpha, self->d_ptr, lda, other->d_ptr, ldb, beta, C->d_ptr, ldc);
          if (cublas_error("cgemm")) return NULL;
        
        } else {
          cublasSgemm(OP(self), OP(other), m, n, k, 1., self->d_ptr, lda, other->d_ptr, ldb, 0., C->d_ptr, ldc);
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
        cuda_DeviceMemory *C = make_vector(m, self->a_dtype);
        if (C == NULL) return NULL;
        int ldc = m;

        if (iscomplex(self)) {
          cuComplex alpha = {1., 0.};
          cuComplex beta = {0., 0.};

          cublasCgemm(OP(self), OP(other), m, n, k, alpha, self->d_ptr, lda, other->d_ptr, ldb, beta, C->d_ptr, ldc);
          if (cublas_error("cgemm")) return NULL;
        
        } else {
          cublasSgemm(OP(self), OP(other), m, n, k, 1., self->d_ptr, lda, other->d_ptr, ldb, 0., C->d_ptr, ldc);
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
        cuda_DeviceMemory *C = make_matrix(m, n, self->a_dtype);
        if (C == NULL) return NULL;
        int ldc = m;

        if (iscomplex(self)) {
          cuComplex alpha = {1., 0.};
          cuComplex beta = {0., 0.};

          cublasCgemm(OP(self), OP(other), m, n, k, alpha, self->d_ptr, lda, other->d_ptr, ldb, beta, C->d_ptr, ldc);
          if (cublas_error("cgemm")) return NULL;
        
        } else {
          cublasSgemm(OP(self), OP(other), m, n, k, 1., self->d_ptr, lda, other->d_ptr, ldb, 0., C->d_ptr, ldc);
          if (cublas_error("sgemm")) return NULL;
        }

        return Py_BuildValue("O", C);
      }
    } 
  }
  else return NULL;
}


/***************************************************************************
 * helper methods for constructing arrays and vectors these are not part of
 * the public api
 ***************************************************************************/

static inline cuda_DeviceMemory*
make_vector(int n, PyArray_Descr *dtype) {

  cuda_DeviceMemory *self = (cuda_DeviceMemory *) _PyObject_New(&cuda_DeviceMemoryType);
  //trace("make_vector: (%d)\n", n);

  if (self != NULL) {
    self->e_size = dtype->elsize;
    self->a_ndims = 1;
    self->a_dims[0] = n;
    self->a_dims[1] = 1; // column vector convention
    self->a_transposed = 0; 
    self->d_ptr = NULL;

    if (cuda_error(cublasAlloc(n, self->e_size, (void**)&self->d_ptr), "make_vector:cublasAlloc")) {
      return NULL;
    }

    self->a_dtype =  dtype;
    Py_INCREF(self->a_dtype);
  }
  return self;
}


static inline cuda_DeviceMemory*
make_matrix(int m, int n, PyArray_Descr *dtype) {

  cuda_DeviceMemory *self = (cuda_DeviceMemory *) _PyObject_New(&cuda_DeviceMemoryType);
  //trace("make_matrix: (%d,%d)\n", m, n);

  if (self != NULL) {
    self->e_size = dtype->elsize;
    self->a_ndims = 2;
    self->a_dims[0] = m;
    self->a_dims[1] = n; 
    self->a_transposed = 0; 
    self->d_ptr = NULL;

    if (cuda_error(cublasAlloc(m*n, self->e_size, (void**)&self->d_ptr), "make_matrix:cublasAlloc")) {
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

    if (PyType_Ready(&cuda_DeviceMemoryType) < 0)
        return;

    module = Py_InitModule3(CUDA_MODULE_NAME, module_methods, "CUDA device memory array module.");

    if (module == NULL) return;
    
    else {
      Py_INCREF(&cuda_DeviceMemoryType);
      PyModule_AddObject(module, CUDA_ARRAY_TYPE_SYM_NAME, (PyObject *)&cuda_DeviceMemoryType);

      cuda_exception = PyErr_NewException(CUDA_ERROR_TYPE_NAME, NULL, NULL);
      Py_INCREF(cuda_exception);
      PyModule_AddObject(module, CUDA_ERROR_TYPE_SYM_NAME, cuda_exception);

      import_array(); // import numpy module
    }
}
