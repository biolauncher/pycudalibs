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

#if defined(_PYCULA_H)
#else
#define _PYCULA_H 1

#include <cula.h>
#if CULA >= 14
#include <cula_scalapack.h>
#endif
#include <pycunumpy.h>
#include <pylibs.h>

/* CULA error handling is more sophisticated than indicated here - TODO take advantage of it!*/
static inline int cula_error(culaStatus status, char* where) {
  trace("CULACALL %s: status = %d\n", where, status);

  // XXX for some reason we are returning without the exception set - check caller code...
  if (status == culaNoError) {
    return 0;

  } else if (status == culaArgumentError) {
    (void) PyErr_Format(cuda_exception, "%s: argument %d has illegal value", where, culaGetErrorInfo());
    return 1;

  } else if (status == culaDataError) {
    (void) PyErr_Format(cuda_exception, "%s: data error with code %d, see LAPACK documentation", where, culaGetErrorInfo());
    return 1;

  } else {
    (void) PyErr_Format(cuda_exception, "%s: %s", where, culaGetStatusString(status));
    return 1;
  }
}

static inline int culablas_error(char* where) {
  return cula_error(culaGetLastStatus(), where);
}

#endif
