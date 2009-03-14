/* copyright (C) 2009 Simon Beaumont - All Rights Reserved */

#define CUDAMEM_MODULE

#include <pycudamem.h>
#include <structmember.h>
#include <stdio.h>

static PyObject* cuda_exception;

// TODO - sort out  the instrumentation and tracing of alloc/dealloc
// TODO - migrate this to pure cuda?

static inline int cuda_error(int status, char* where) {

  trace("CUDACALL %s: status = %d\n", where, status);

  if (status == CUBLAS_STATUS_SUCCESS) {
    return 0;

  } else {
    PyErr_SetString(cuda_exception, get_cublas_error_text(status));
    return 1;
  }
}

static void
cuda_DeviceMemory_dealloc(cuda_DeviceMemory* self) {

  /* don't really understand what this callback is
  PyObject *cbresult;

  if (self->my_callback != NULL) {
    PyObject *err_type, *err_value, *err_traceback;
    int have_error = PyErr_Occurred() ? 1 : 0;
    
    if (have_error)
      PyErr_Fetch(&err_type, &err_value, &err_traceback);
    
    cbresult = PyObject_CallObject(self->my_callback, NULL);
    if (cbresult == NULL)
      PyErr_WriteUnraisable(self->my_callback);
    else
      Py_DECREF(cbresult);
    
    if (have_error)
      PyErr_Restore(err_type, err_value, err_traceback);
    
    Py_DECREF(self->my_callback);
  } */

  trace("TRACE DeviceMemory_dealloc: %0x (%0x)\n", (int) self, (int) self->d_ptr);

  if (self->d_ptr != NULL)  {
    if (cuda_error(cublasFree(self->d_ptr), "dealloc:cublasFree"))
      return;
    else
      self->ob_type->tp_free((PyObject*)self);
  }
}


static PyObject *
cuda_DeviceMemory_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  cuda_DeviceMemory *self;

  self = (cuda_DeviceMemory *)type->tp_alloc(type, 0);
    
  if (self != NULL) {
    self->d_ptr = NULL;
    self->e_size = 0;
    self->e_num = 0;
  }
  
  return (PyObject *)self;
}

static int
cuda_DeviceMemory_init(cuda_DeviceMemory *self, PyObject *args, PyObject *kwds) {

  if (! PyArg_ParseTuple(args, "ii", &self->e_num, &self->e_size))
    return -1; 
  
  trace("elements: %d element_size: %d\n", self->e_num, self->e_size);
    
  if (self->d_ptr != NULL && cuda_error(cublasFree(self->d_ptr), "init:cublasFree"))
    return -1;
  
  if (cuda_error(cublasAlloc(self->e_num, self->e_size, (void**)&self->d_ptr), "init:cublasAlloc"))
    return -1;
  
  return 0;
}

/* pretty print etc. */

static inline PyObject *
_stringify(cuda_DeviceMemory *self) {
  return PyString_FromFormat("<%s (0x%0x: %d x %d) @0x%0x>",
                             self->ob_type->tp_name,
                             (int)self->d_ptr,
                             self->e_num,
                             self->e_size,
                             (int)self);
}

static PyObject *
cuda_DeviceMemory_repr(cuda_DeviceMemory *self) {
  return _stringify(self);
}

static PyObject *
cuda_DeviceMemory_str(cuda_DeviceMemory *self) {
  return _stringify(self);
}


/* N.B. we don't expose the device pointer at all 
   TODO some nice class constants for double and float sizes */

PyMemberDef cuda_DeviceMemory_members[] = {
    {"elements", T_INT, offsetof(cuda_DeviceMemory, e_num), READONLY,
     "number of elements"},
    {"element_size", T_INT, offsetof(cuda_DeviceMemory, e_size), READONLY,
     "size of each element"},
    {NULL}
};


/******************************************************
 * device memory loads and stores - no bounds checking 
 ******************************************************/

/*
cublasStatus 
cublasGetVector (int n, int elemSize, const void *x, 
 int incx, void *y, int incy) 

copies n elements from a vector x in GPU memory space to a vector y  
in CPU memory space. Elements in both vectors are assumed to have a  
size of elemSize bytes. Storage spacing between consecutive elements  
is incx for the source vector x and incy for the destination vector y. In  
general, x points to an object, or part of an object, allocated via  
cublasAlloc(). Column‐major format for two‐dimensional matrices  
is assumed throughout CUBLAS. If the vector is part of a matrix, a  
vector increment equal to 1 accesses a (partial) column of the matrix.  
Similarly, using an increment equal to the leading dimension of the  
matrix accesses a (partial) row.  
*/

static PyObject*
cuda_DeviceMemory_getVector(cuda_DeviceMemory *self, PyObject *args) {
  int n, elemSize, incx, incy, vsize;
  void* vector;

  if (PyArg_ParseTuple(args, "iiit#i", &n, &elemSize, &incx, &vector, &vsize, &incy))
    if (cuda_error(cublasGetVector(n, elemSize, self->d_ptr, incx, vector, incy), "cublasGetVector"))
      return NULL;
    else
      return Py_BuildValue("");
  else
    return NULL;
}


/*
cublasStatus 
cublasSetVector (int n, int elemSize, const void *x, 
                 int incx, void *y, int incy) 

copies n elements from a vector x in CPU memory space to a vector y  
in GPU memory space. Elements in both vectors are assumed to have a  
size of elemSize bytes. Storage spacing between consecutive elements  
is incx for the source vector x and incy for the destination vector y. In  
general, y points to an object, or part of an object, allocated via  
cublasAlloc(). Column‐major format for two‐dimensional matrices  
is assumed throughout CUBLAS. If the vector is part of a matrix, a  
vector increment equal to 1 accesses a (partial) column of the matrix.  
*/

static PyObject*
cuda_DeviceMemory_setVector(cuda_DeviceMemory *self, PyObject *args) {
  int n, elemSize, incx, incy, vsize;
  void* vector;

  if (PyArg_ParseTuple(args, "iiit#i", &n, &elemSize, &incx, &vector, &vsize, &incy))
    if (cuda_error(cublasSetVector(n, elemSize, self->d_ptr, incx, vector, incy), "cublasSetVector"))
      return NULL;
    else
      return Py_BuildValue("");
  else
    return NULL;
}



/***************
 * method table
 **************/

static PyMethodDef cuda_DeviceMemory_methods[] = {
  {"storeVector", (PyCFunction) cuda_DeviceMemory_getVector, METH_VARARGS,
   "Store DeviceMemory into host memory."},
  {"loadVector", (PyCFunction) cuda_DeviceMemory_setVector, METH_VARARGS,
   "Load host memory into DeviceMemory."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};


/**************
 * object type
 **************/

static PyTypeObject cuda_DeviceMemoryType = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /*ob_size*/
    "cuda.DeviceMemory",                      /*tp_name*/
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
    "CUDA device memory references",          /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    cuda_DeviceMemory_methods,                /* tp_methods */
    cuda_DeviceMemory_members,                /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)cuda_DeviceMemory_init,         /* tp_init */
    0,                                        /* tp_alloc */
    cuda_DeviceMemory_new,                    /* tp_new */
};

 static PyMethodDef module_methods[] = {
  {NULL}  /* Sentinel */
};


#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_cudamem(void) {
    PyObject* module;

    if (PyType_Ready(&cuda_DeviceMemoryType) < 0)
        return;

    module = Py_InitModule3("_cudamem", module_methods, "CUDA device memory utility module.");

    if (module == NULL) return;
    
    else {
      Py_INCREF(&cuda_DeviceMemoryType);
      PyModule_AddObject(module, "DeviceMemory", (PyObject *)&cuda_DeviceMemoryType);

      cuda_exception = PyErr_NewException("cudamem.error", NULL, NULL);
      Py_INCREF(cuda_exception);
      PyModule_AddObject(module, "error", cuda_exception);
    }
}
