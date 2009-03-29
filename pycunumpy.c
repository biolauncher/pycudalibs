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


  if (PyArg_ParseTuple(args, "O", &object)) {

    array = (PyArrayObject*) PyArray_FROM_OTF(object, NPY_FLOAT32, NPY_FORTRAN | NPY_ALIGNED);
    if (array == NULL) return NULL;
    
    else {
      // create DeviceMemory

      npy_intp *dims = PyArray_DIMS(array);
      int ndims = PyArray_NDIM(array);

      devicemem = (cuda_DeviceMemory*) cudamem_DeviceMemoryType->tp_new(cudamem_DeviceMemoryType, NULL, NULL);
      //devicemem = (cuda_DeviceMemory*) PyType_GenericNew(cudamem_DeviceMemoryType, NULL, NULL);
        
      if (devicemem != NULL) {

        if (PyObject_CallMethod((PyObject *) devicemem, "__init__", "iiii", 
                                ndims, dims[0], dims[1], FLOAT32_BYTES) == NULL) {
          Py_DECREF(array);
          Py_DECREF(devicemem);
          return NULL;
        }

        void* source = PyArray_DATA(array);
        int n = a_elements(devicemem);
        
        if (cuda_error(cublasSetVector(n, devicemem->e_size, source, 1, 
                                       devicemem->d_ptr, 1), "cublasSetVector")) {
          Py_DECREF(array);
          Py_DECREF(devicemem);
          return NULL;
        }

        Py_DECREF(array);
        return Py_BuildValue("N", devicemem);
          
      } else {
        Py_DECREF(array);
        return NULL;
      }
    }
  } else return NULL;
}

/************************************
 * cuda device memory to numpy array
 ************************************/

static inline int toNumpyType(int flags) {
  return (flags & COMPLEX_TYPE) ?
    ((flags & DOUBLE_TYPE) ? PyArray_COMPLEX128 : PyArray_COMPLEX64) :
    ((flags & DOUBLE_TYPE) ? PyArray_FLOAT64 : PyArray_FLOAT32);
}

static PyObject*
cuda_DeviceMemory2numpy_Array(PyObject *dummy, PyObject *args) {
  PyObject *object;

  if (PyArg_ParseTuple(args, "O!", cudamem_DeviceMemoryType, &object)) {
    cuda_DeviceMemory *deviceMemory = (cuda_DeviceMemory *) object;

    npy_intp dims[2];
    dims[0] = deviceMemory->a_dims[0];
    dims[1] = deviceMemory->a_dims[1];

    // create a numpy array to hold the data
    PyObject *array = PyArray_EMPTY(deviceMemory->a_ndims, dims, 
                                    //toNumpyType(deviceMemory->a_flags),
                                    PyArray_FLOAT32,
                                    1);
    if (array != NULL) {
      // fill it in
      if (cuda_error(cublasGetVector (a_elements(deviceMemory), deviceMemory->e_size, 
                                      deviceMemory->d_ptr, 1, PyArray_DATA(array), 1),
                     "cublasGetVector")) {
        
        Py_DECREF(array);
        return NULL;
      
      } else return Py_BuildValue("N", array);
    }
  } 
  return NULL; 
}

/*************************
 * numpy extension module
 *************************/

static PyMethodDef functions[] = {
  {"array_to_cuda", (PyCFunction) numpy_Array2cuda_DeviceMemory, METH_VARARGS,
   "Load an array to CUDA device."},
  {"cuda_to_array", (PyCFunction) cuda_DeviceMemory2numpy_Array, METH_VARARGS,
   "Store an array from CUDA device."},
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
