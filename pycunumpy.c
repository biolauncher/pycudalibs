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
  PyObject *devicemem = NULL;
  int rank;


  if (PyArg_ParseTuple(args, "O", &object)) {
    trace("parsed args\n");
    array = (PyArrayObject*)PyArray_FROM_OTF(object, NPY_FLOAT32, NPY_FORTRAN | NPY_ALIGNED);
    trace("got array\n");
    
    if (array == NULL) return NULL;
    
    else {

      // check dimensions or rank - we only do vectors and matrices.
      rank = PyArray_NDIM(array);
      if (rank < 1 || rank > 2) {
        // TODO create exception

      } else {
        trace("dims ok\n");
        // create DeviceMemory
        npy_intp *dims = PyArray_DIMS(array);
        //int elements = (rank == 1) ? dims[0] : dims[0] * dims[1];

        devicemem = PyType_GenericNew(cudamem_DeviceMemoryType, NULL, NULL);
        trace("new device mem\n");
        
        if (devicemem != NULL) {
          (void) PyObject_CallMethod(devicemem, "__init__", "iiii", rank, dims[0], dims[1], FLOAT32_BYTES);
          trace("init device mem\n");

          // fill it in hmmm need to sort this method
          // int elements, int elem-size, void* data
          //PyObject* PyObject_CallMethodObjArgs(PyObject *o, PyObject *name, ..., NULL);
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
