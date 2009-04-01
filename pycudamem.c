/* copyright (C) 2009 Simon Beaumont - All Rights Reserved */

#define CUDAMEM_MODULE
#include <pycudamem.h>
#include <structmember.h>
#include <stdio.h>


static void
cuda_DeviceMemory_dealloc(cuda_DeviceMemory* self) {

  trace("TRACE cuda_DeviceMemory_dealloc: %0x (%0x)\n", (int) self, (int) self->d_ptr);

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
    self->a_ndims = 0;
    self->a_dims[0] = 0;
    self->a_dims[1] = 0;
    self->a_flags = 0; 
  }
  
  return (PyObject *)self;
}

static int
cuda_DeviceMemory_init(cuda_DeviceMemory *self, PyObject *args, PyObject *kwds) {

  if (! PyArg_ParseTuple(args, "iiii", &self->a_ndims, &self->a_dims[0], &self->a_dims[1], &self->e_size))
    return -1; 

  // may lift this in time
  if (self->a_ndims < 1 || self->a_ndims > 2) {
    PyErr_SetString(PyExc_TypeError, "number of array dimensions must be 1 or 2");
    return -1;
  }

  int n_elements = a_elements(self);

  trace("TRACE cuda_DeviceMemory_init: elements: %d element_size: %d\n", n_elements, self->e_size);
    
  if (self->d_ptr != NULL && cuda_error(cublasFree(self->d_ptr), "init:cublasFree"))
    return -1;
  
  if (cuda_error(cublasAlloc(n_elements, self->e_size, (void**)&self->d_ptr), "init:cublasAlloc"))
    return -1;
  
  return 0;
}

/* pretty print etc. */

static inline PyObject *
_stringify(cuda_DeviceMemory *self) {
  return PyString_FromFormat("<%s (0x%0x: %d X %d  %s %s (%d) %s) @0x%0x>",
                             self->ob_type->tp_name,
                             (int)self->d_ptr,
                             self->a_dims[0],
                             self->a_dims[1],
                             (self->a_flags & COMPLEX_TYPE) ? "complex" : "real",
                             (self->a_flags & DOUBLE_TYPE) ? "double" : "single",
                             self->e_size,
                             (self->a_ndims == 2) ? "matrix" : (self->a_ndims == 1 ? "vector" : "scalar"),
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

/* accessor methods */

static PyObject*
cuda_DeviceMemory_getShape(cuda_DeviceMemory *self, void *closure) {
  if (self->a_ndims == 2) return Py_BuildValue("(ii)", self->a_dims[0], self->a_dims[1]);
  else return Py_BuildValue("(i)", self->a_dims[0]);
}

/***********************************
 * expose basic informational slots 
 ***********************************/

static PyMemberDef cuda_DeviceMemory_members[] = {
  {"element_size", T_INT, offsetof(cuda_DeviceMemory, e_size), READONLY,
   "Size of each device array element"},
  {NULL}
};


/***************
 * method table
 **************/

static PyMethodDef cuda_DeviceMemory_methods[] = {
  /*
    {"shape", (PyCFunction) cuda_DeviceMemory_getShape, METH_VARARGS,
    "Get the shape of the device memory."},
  */
  {NULL, NULL, 0, NULL} 
};

/**********************
 * getters and setters
 *********************/
static PyGetSetDef cuda_DeviceMemory_properties[] = {
  {"shape", (getter) cuda_DeviceMemory_getShape, (setter) NULL, 
   "shape of device array", NULL},
  // TODO numpy style dtype?
  {NULL}
};


/**************
 * object type
 **************/

static PyTypeObject cuda_DeviceMemoryType = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /*ob_size*/
    CUDAMEM_ARRAY_TYPE_NAME,                  /*tp_name*/
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
    "CUDA device array",                      /* tp_doc */
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

    module = Py_InitModule3(CUDAMEM_MODULE_NAME, module_methods, "CUDA device memory utility module.");

    if (module == NULL) return;
    
    else {
      Py_INCREF(&cuda_DeviceMemoryType);
      PyModule_AddObject(module, CUDAMEM_ARRAY_TYPE_NAME, (PyObject *)&cuda_DeviceMemoryType);

      cuda_exception = PyErr_NewException(CUDAMEM_ERROR_TYPE_NAME, NULL, NULL);
      Py_INCREF(cuda_exception);
      PyModule_AddObject(module, CUDAMEM_ERROR_TYPE_NAME, cuda_exception);
    }
}
