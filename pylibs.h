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


/* 
 * General macros and definitions
 */

#if defined(_PYLIBS_H)
#else
#define _PYLIBS_H

#define max(a,b) (a)>(b)?(a):(b)
#define min(a,b) (a)<(b)?(a):(b)

#if DEBUG > 0
#warning "N.B. tracing code will be compiled in"
#include <stdio.h>
#define trace(format, ...) fprintf(stderr, format, ## __VA_ARGS__)
#else
#define trace(format, ...)
#endif

#endif
