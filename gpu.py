#
# Copyright (C) 2009 Model Sciences Ltd.
#
# This file is  part of pycudalibs
#
#    pycudalibs is free software: you can redistribute it and/or modify
#    it under the terms of the Lesser GNU General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    Pycudalibs is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the Lesser GNU General Public
#    License along with pycudalibs.  If not, see <http://www.gnu.org/licenses/>.  

import _cunumpy
import _cublas

class CUDAdevice (object):

    def __init__(self, device_number=0):
        _cublas.select_device(device_number)
        _cublas.init()

    def __enter__(self):
        return self

    # currently this closes down the library! which seems to
    # cause a problem at ipython shutdown (as package symbols are deleted?)
    # also a thread may only be bound to one cuda device XXX review in light of CUDA4
    # so the "with CUDAdevice" is not such a useful idiom right now. but I am leaving this
    # "bug" in for now as it does not cause a problem with regular python.
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    @staticmethod
    def device_count():
        return _cublas.device_count()

    def close(self):
        _cublas.close()

#
# extend _cunumpy.array 
#
class CUDAarray(_cunumpy.array):
    """
    Encapsulates CUDA device based arrays numpy style.
    """
    pass
