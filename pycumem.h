/* copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/* 
 * a python object which provides a simple encapsulation of CUDA device
 * memory that can be shared. 
 *
 * Currently a C only api that can be included in other modules
 */

#if defined(_PYCUMEM_H)
#else
#define _PYCUMEM_H 1

#define CUDA_MEMORY_TYPE_NAME "cuda.memory"


typedef struct {
  PyObject_HEAD
  void* d_ptr;                       /* opaque device pointer */
} cuda_Memory;


/**
 * The raison d'etre of this object is to do referenced counted de-allocation
 */
static void
cuda_Memory_dealloc(cuda_Memory* self) {

  trace("TRACE cuda_Memory_dealloc: %0x (%0x)\n", (int) self, (int) self->d_ptr);

  self->ob_type->tp_free((PyObject*)self);

  if (self->d_ptr != NULL) 
    if (cuda_error(cublasFree(self->d_ptr), "dealloc:cublasFree"))
      return;
}

/**
 * allocate a new object
 */
static PyObject *
cuda_Memory_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  cuda_Memory *self;

  self = (cuda_Memory *)type->tp_alloc(type, 0);
    
  if (self != NULL) {
    self->d_ptr = NULL;
  }
  
  return (PyObject *)self;
}

/**************
 * object type
 **************/

static PyTypeObject cuda_MemoryType = {
    PyObject_HEAD_INIT(NULL)
    0,                                        /* ob_size*/
    CUDA_MEMORY_TYPE_NAME,                    /* tp_name*/
    sizeof(cuda_Memory),                      /* tp_basicsize*/
    0,                                        /* tp_itemsize*/
    (destructor)cuda_Memory_dealloc,          /* tp_dealloc*/
    0,                                        /* tp_print*/
    0,                                        /* tp_getattr*/
    0,                                        /* tp_setattr*/
    0,                                        /* tp_compare*/
    0,                                        /* tp_repr*/
    0,                                        /* tp_as_number*/
    0,                                        /* tp_as_sequence*/
    0,                                        /* tp_as_mapping*/
    0,                                        /* tp_hash */
    0,                                        /* tp_call*/
    0,                                        /* tp_str*/
    0,                                        /* tp_getattro*/
    0,                                        /* tp_setattro*/
    0,                                        /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags*/
    "CUDA device memory",                     /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    cuda_Memory_new,                          /* tp_new */
};

/******************************
 * c utility factory functions
 ******************************/

/**
 * create a new object and the required memory to manage
 */
static inline cuda_Memory*
alloc_cuda_Memory(int n, int size) {
  
  cuda_Memory *self = (cuda_Memory *) _PyObject_New(&cuda_MemoryType);
  
  if (self != NULL) {
    self->d_ptr = NULL;

    if (cuda_error(cublasAlloc(n, size, (void**) &self->d_ptr), "cuda_Memory:cublasAlloc")) {
      return NULL;
    }
  }
  return self;
}

/**
 * embed this in client module init
 */ 
static inline int 
init_cuda_MemoryType(void) {
  return PyType_Ready(&cuda_MemoryType);
}


#endif


