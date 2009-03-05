/* copyright (C) 2009 Simon Beaumont - All Rights Reserved */

#include <pycuda.h>
#include <structmember.h>
#include <stdio.h>

static PyObject* cuda_exception;

static inline int cuda_error(int status, char* where) {
#if DEBUG > 0
  fprintf(stderr, "CUDA %s: status = %d\n", where, status);
#endif
  if (status == CUBLAS_STATUS_SUCCESS) {
    return 0;

  } else {
    PyErr_SetString(cuda_exception, get_cublas_error_text(status));
    return 1;
  }
}

static void
cuda_DeviceMemory_dealloc(cuda_DeviceMemory* self) {
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

    static char *kwlist[] = {"elements", "element_size", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, 
                                      &self->e_num,
                                      &self->e_size))
      return -1; 

    else if (self->d_ptr != NULL && cuda_error(cublasFree(self->d_ptr), "init:cublasFree"))
      return -1;
    
    else if (cuda_error(cublasAlloc(self->e_num, self->e_size, (void**)&self->d_ptr), "init:cublasAlloc"))
      return -1;

    else
      return 0;
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



static PyMethodDef cuda_DeviceMemory_methods[] = {
  {NULL}  // not yet rosin, not yet...
};


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
    0,                                        /*tp_repr*/
    0,                                        /*tp_as_number*/
    0,                                        /*tp_as_sequence*/
    0,                                        /*tp_as_mapping*/
    0,                                        /*tp_hash */
    0,                                        /*tp_call*/
    0,                                        /*tp_str*/
    0,                                        /*tp_getattro*/
    0,                                        /*tp_setattro*/
    0,                                        /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "DeviceMemory objects",                   /* tp_doc */
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
init_cuda(void) 
{
    PyObject* module;

    if (PyType_Ready(&cuda_DeviceMemoryType) < 0)
        return;

    module = Py_InitModule3("_cuda", module_methods,"CUDA utility module.");

    if (module == NULL) return;
    
    else {
      Py_INCREF(&cuda_DeviceMemoryType);
      PyModule_AddObject(module, "DeviceMemory", (PyObject *)&cuda_DeviceMemoryType);

      cuda_exception = PyErr_NewException("cuda.error", NULL, NULL);
      Py_INCREF(cuda_exception);
      PyModule_AddObject(module, "error", cuda_exception);
    }
}
