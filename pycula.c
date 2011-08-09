/*
Copyright (C) 2009 Model Sciences Ltd.

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
*/

 
/**
 * Python integration to CULA framework routines.
 *  defines module: _cula 
 */

#include <pycula.h>

/**
 * get number of CUDA devices
 */
static PyObject* getDeviceCount(PyObject* self, PyObject* args) {
  int n_dev;

  if (!PyArg_ParseTuple(args, "")) 
    return NULL;
  else if (cula_error(culaGetDeviceCount(&n_dev), "culaGetDeviceCount"))
    return NULL;
  else 
    return Py_BuildValue("i", n_dev);
}

/**
 * select a cuda device to bind to this thread - must be called
 * before init.
 */
static PyObject* selectDevice(PyObject* self, PyObject* args) {
  int devno;

  if (!PyArg_ParseTuple(args, "i", &devno)) 
    return NULL;
  else if (cula_error(culaSelectDevice(devno), "culaSelectDevice"))
    return NULL;
  else 
    return Py_BuildValue("");

}

static PyObject* init(PyObject* self, PyObject* args) {
  if (!PyArg_ParseTuple(args, "")) 
    return NULL;
  else if (cula_error(culaInitialize(), "culaInitialize"))
    return NULL;
  else 
    return Py_BuildValue("");
}

static PyObject* shutdown(PyObject* self, PyObject* args) {
  if (!PyArg_ParseTuple(args, "")) 
    return NULL;
  // culaShutdown doesn't return any value
  else {
    culaShutdown();
    return Py_BuildValue("");
  }
}


/************************
 * module function table
 ************************/

static PyMethodDef _cula_methods[] = {

  {"device_count", getDeviceCount, METH_VARARGS,
   "Get the number of CUDA capable devices."},

  {"select_device", selectDevice, METH_VARARGS,
   "Select the CUDA device to attach to the host thread."},

  {"init", init, METH_VARARGS, 
   "Initialise CULA library by attaching to CUDA device that is bound to the calling host thread."},

  {"close", shutdown, METH_VARARGS,
   "Shutdown the CULA library."},
  {NULL, NULL, 0, NULL}
};


/* initialise the Python c extension module - this function has to be named consistently */

PyMODINIT_FUNC init_cula(void) {
  // initialise the module
  PyObject* module = Py_InitModule("_cula", _cula_methods);
  if (module == NULL) return;
}

