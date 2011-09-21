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

import _cula

class device (object):

    def __init__(self, device_number=0):
        self.device_number = device_number
        _cula.select_device(device_number)
        _cula.init()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    @staticmethod
    def device_count():
        return _cula.device_count()

    def close(self):
        _cula.close()

    def reset(self):
        _cula.reset()

    @staticmethod
    def shutdown():
        _cula.shutdown()

