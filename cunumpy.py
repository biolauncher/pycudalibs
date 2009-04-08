# Copyright (c) 2009 Simon Beaumont <seb@modelsciences.com> - All Rights Reserved

import _cunumpy
import numpy

#
# module of functions to provide numpy like factory methods for _cunumpy.array objects
#

def array(*args, **keyw):
    return _cunumpy.array(numpy.array(*args, **keyw))




