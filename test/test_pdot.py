import unittest
import cunumpy as cn
import numpy as np
import math
import test

class TestMultiGPUDotProduct(unittest.TestCase):
    
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

    # vector-vector dot products
    def test_real_single_vector_vector_dot(self):

        na = np.array(self.real_veca, dtype=np.float32)
        nb = np.array(self.real_vecb, dtype=np.float32)
        nc = np.dot(na,nb)
        nd = cn.pdot(na, nb)
        self.assert_(test.scalars_equal(nc, nd, epsilon=0.05))


    def test_complex_single_vector_vector_dot(self):
        va = cn.array(self.complex_veca, dtype=cn.complex64)
        vb = cn.array(self.complex_vecb, dtype=cn.complex64)
        vc = va.dot(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_veca, dtype=np.complex64)
        nb = np.array(self.complex_vecb, dtype=np.complex64)
        nc = np.dot(na,nb)
        self.assert_(test.scalars_equal(vc, nc, epsilon=0.05))


    # vector-matrix dot products
    def test_real_single_vector_matrix_dot(self):
        na = np.array(self.real_veca, dtype=np.float32)
        nb = np.array(self.real_matb, dtype=np.float32)
        nc = np.dot(na,nb)
        nd = cn.pdot(na, nb)
        self.assert_(test.arrays_equal(nd, nc, epsilon=0.05))


    def test_complex_single_vector_matrix_dot(self):
        pass

    def test_complex_double_vector_matrix_dot(self):
        pass

    # matrix-vector dot products
    def test_real_single_matrix_vector_dot(self):
        na = np.array(self.real_mata, dtype=np.float32)
        nb = np.array(self.real_vecb, dtype=np.float32)
        nc = np.dot(na,nb)
        nd = cn.pdot(na, nb)
        self.assert_(test.arrays_equal(nd, nc, epsilon=0.05))


    def test_real_double_matrix_vector_dot(self):
        pass

    def test_complex_single_matrix_vector_dot(self):
        pass

    def test_complex_double_matrix_vector_dot(self):
        pass

    # matrix-matrix dot products
    def test_real_single_matrix_matrix_dot(self):
        na = np.array(self.real_mata, dtype=np.float32)
        nb = np.array(self.real_matb, dtype=np.float32)
        nc = np.dot(na,nb)
        nd = cn.pdot(na,nb)
        self.assert_(test.arrays_equal(nd, nc, epsilon=0.05))


    def test_real_double_matrix_matrix_dot(self):
        pass

    def test_complex_single_matrix_matrix_dot(self):
        pass

    def test_complex_double_matrix_matrix_dot(self):
        pass


def suite_single():
    suite = unittest.TestSuite()
    tests = ['test_real_single_vector_vector_dot',
             #'test_complex_single_vector_vector_dot',
             'test_real_single_vector_matrix_dot',
             # not in CUBLAS 'test_complex_single_vector_matrix_dot',
             'test_real_single_matrix_vector_dot',
             # not in CUBLAS 'test_complex_single_matrix_vector_dot',
             'test_real_single_matrix_matrix_dot',
             #'test_complex_single_matrix_matrix_dot'
             ]

    return unittest.TestSuite(map(TestMultiGPUDotProduct, tests))

def suite_double():
    suite = unittest.TestSuite()
    tests = ['test_real_double_vector_vector_dot',
             'test_complex_double_vector_vector_dot',
             'test_real_double_vector_matrix_dot',
             'test_complex_double_vector_matrix_dot',
             'test_real_double_matrix_vector_dot',
             'test_complex_double_matrix_vector_dot',
             'test_real_double_matrix_matrix_dot',
             'test_complex_double_matrix_matrix_dot']

    return unittest.TestSuite(map(TestMultiGPUDotProduct, tests))


def suite():
    return unittest.TestSuite(suite_single())

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
