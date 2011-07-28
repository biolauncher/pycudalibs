import unittest
import cunumpy as cn
import numpy as np
import math
import test

class TestTranspose(unittest.TestCase):
    
    def setUp(self):
        self.real_veca = [3,4,0,11.,12,15.1,-2.,9.,14.,16.7]
        self.real_vecb = [7.0003,4.1,9.7,1.03,1.1210008,1.1500013,-2.9876,9.0002,3005.01,22.000007]
        self.complex_veca = [1, 0, 2+4j, -1-math.pi*1j, 2.-3.7j, 5., 3.j, -2.3j, -math.e+math.e*1j, 7]
        self.complex_vecb = [1, 0, 12-4.09j, -5.00002+math.pi*1j, 
                             22.-13.077j, 5., 2.003+3.004j, -2.3j, 1j, 0.0007j]
        self.real_mata = [self.real_veca for i in range(1,5)]
        self.real_matb = [self.real_vecb for i in range(1,11)]
        self.complex_mata = [self.complex_veca for i in range(1,5)]
        self.complex_matb = [self.complex_vecb for i in range(1,11)]

    def test_vector_transpose_identity(self):
        # transpose is a noop for vectors - they are always columns
        v = cn.array(self.real_veca, dtype=cn.float32)
        self.assert_(test.arrays_equal(v.T.toarray(), v.toarray()))

    def test_matrix_transpose_identity(self):
        a = cn.array(self.real_mata, dtype=cn.float32)
        self.assert_(test.arrays_equal(a.toarray(), a.T.T.toarray()))

    def test_linalg_transpose_identity(self):
        a = cn.array(self.real_mata, dtype=cn.float32)
        b = cn.array(self.real_matb, dtype=cn.float32)
        c = a.dot(b)
        d = b.T.dot(a.T).T
        self.assert_(test.arrays_equal(c.toarray(), d.toarray(), epsilon=0.05))


def suite_single():
    suite = unittest.TestSuite()
    tests = ['test_vector_transpose_identity',
             'test_matrix_transpose_identity',
             'test_linalg_transpose_identity'
             ]

    return unittest.TestSuite(map(TestTranspose, tests))

def suite_double():
    suite = unittest.TestSuite()
    tests = []
    return unittest.TestSuite(map(TestTranspose, tests))

def suite():
    return unittest.TestSuite(suite_single())

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
