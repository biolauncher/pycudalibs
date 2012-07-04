import sys
import unittest
import cunumpy as cn
import gpu
import numpy as np
import math
from _cunumpy import CUDAERROR
import test

# we will be testing linear algebra on CUDA so we need an approximation YMMV
def arrays_equal(a, b, epsilon=0.000001):
    return (abs(a-b) < epsilon).all()

def scalars_equal(a, b, epsilon=0.00001):
    return abs(a-b) < epsilon

class TestMemory(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_1024x1024(self):
        a = cn.array(np.random.rand(1024,1204), dtype=cn.float32)
    
    def test_2048x2048(self):
        a = cn.array(np.random.rand(2048,2048), dtype=cn.float32)
    
    def test_4096x4096(self):
        a = cn.array(np.random.rand(4096,4096), dtype=cn.float32)
    
    def test_8192x8192(self):
        a = cn.array(np.random.rand(8192,8192), dtype=cn.float32)
    
    def test_8192x8192_dot(self):
        a = cn.array(np.random.rand(8192,8192), dtype=cn.float32)
        b = a.dot(a)

    def test_4096x4096_dot(self):
        a = cn.array(np.random.rand(4096,4096), dtype=cn.float32)
        b = a.dot(a)

    def test_maxmem(self):
        k = 0
        try:
            while k < 500:
                k += 1
                n = k * 1024 * 1024
                a = cn.array(np.random.rand(n), dtype=cn.float32)
                print k * 4, ' ',
                sys.stdout.flush()
        except CUDAERROR, e:
            print e, '@', k * 4, 'MB ',
            sys.stdout.flush()
            #self._assert(True)

def suite_single():
    suite = unittest.TestSuite()
    tests = [
        'test_1024x1024',
        'test_2048x2048',
        'test_4096x4096',
        'test_8192x8192',
        #'test_8192x8192_dot',
        'test_4096x4096_dot',
        'test_maxmem'
        ]

    return unittest.TestSuite(map(TestMemory, tests))


def suite():
    return unittest.TestSuite(suite_single())

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
