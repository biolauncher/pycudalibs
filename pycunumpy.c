/* Copyright (c) 2009 Simon Beaumont - All Rights Reserved */

/**
 * numpy integration with cudamemory
 */

#include <pycudamem.h>
#include <arrayobject.h>


/************************************
 * numpy array to cuda device memory
 *  take a numpy array 
 *  return a cuda DeviceMemory object 
 ************************************/
static PyObject*
numpy_Array2cuda_DeviceMemory(PyObject *dummy, PyObject *args) {
  PyObject *object = NULL;
  PyArrayObject *array = NULL;
  cuda_DeviceMemory *devicemem = NULL;
  int ndims;


  if (PyArg_ParseTuple(args, "O", &object)) {
    trace("parsed args\n");
    array = (PyArrayObject*) PyArray_FROM_OTF(object, NPY_FLOAT32, NPY_FORTRAN | NPY_ALIGNED);
    trace("got array\n");
    
    if (array == NULL) return NULL;
    
    else {

      // check dimensions or ndims - we only do vectors and matrices.
      ndims = PyArray_NDIM(array);
      if (ndims < 1 || ndims > 2) {
        PyErr_SetString(PyExc_TypeError, "number of array dimensions must be 1 or 2");

      } else {
        trace("dims ok\n");
        // create DeviceMemory
        npy_intp *dims = PyArray_DIMS(array);

        devicemem = (cuda_DeviceMemory*) PyType_GenericNew(cudamem_DeviceMemoryType, NULL, NULL);
        trace("new device mem\n");
        
        if (devicemem != NULL) {
          (void) PyObject_CallMethod((PyObject *) devicemem, "__init__", "iiii", 
                                     ndims, dims[0], dims[1], FLOAT32_BYTES);
          trace("init device mem\n");

          void* source = PyArray_DATA(array);
          int n = a_elements(devicemem);

          // would set matrix be quicker here for matrices?
          if (cuda_error(cublasSetVector(n, devicemem->e_size, source, 1, 
                                         devicemem->d_ptr, 1), "cublasSetVector")) {
            Py_DECREF(array);
            return NULL;
          }

          Py_DECREF(array);
          trace("decref array\n");
          return Py_BuildValue("N", devicemem);
          
        } else {
          trace("null devicemem\n");
          Py_DECREF(array);
          return NULL;
        }
      }
      Py_DECREF(array);
      return NULL;
    }
  } else return NULL;
}

/************************************
 * cuda device memory to numpy array
 ************************************/
static PyObject*
cuda_DeviceMemory2numpy_Array(PyObject *dummy, PyObject *args) {
  PyObject *object;

  if (PyArg_ParseTuple(args, "O!", cudamem_DeviceMemoryType, &object)) {
    //cuda_DeviceMemory *deviceMemory = (cuda_DeviceMemory *) object;
    // create a numpy array to hold the data
  } else return NULL; 

  //int PyObject_IsInstance(PyObject *inst, PyObject *cls)
}

/*************************
 * numpy extension module
 *************************/

static PyMethodDef functions[] = {
  {"array_to_cuda", (PyCFunction) numpy_Array2cuda_DeviceMemory, METH_VARARGS,
   "Load an array to CUDA device."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_cunumpy(void) {
  PyObject* module;
  module = Py_InitModule("_cunumpy", functions);
  // load numpy
  import_array();
  // load _cudamem module
  import_cudamem();
}
