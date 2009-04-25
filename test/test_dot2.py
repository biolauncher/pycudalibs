import unittest
import cunumpy as cn
import numpy as np
import math

# we will be testing linear algebra on CUDA we we need an approximation
def arrays_equal(a, b, epsilon=0.000001):
    return (abs(a-b) < epsilon).all()

def scalars_equal(a, b, epsilon=0.00001):
    return abs(a-b) < epsilon

class TestDotProduct2(unittest.TestCase):

    
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

    # vector-vector dot2 products
    def test_real_single_vector_vector_dot2(self):
        va = cn.array(self.real_veca, dtype=cn.float32)
        vb = cn.array(self.real_vecb, dtype=cn.float32)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.real_veca, dtype=np.float32)
        nb = np.array(self.real_vecb, dtype=np.float32)
        nc = np.dot(na,nb)
        self.assert_(scalars_equal(vc, nc, epsilon=0.05))

    def test_real_double_vector_vector_dot2(self):
        va = cn.array(self.real_veca, dtype=cn.float64)
        vb = cn.array(self.real_vecb, dtype=cn.float64)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.real_veca, dtype=np.float64)
        nb = np.array(self.real_vecb, dtype=np.float64)
        nc = np.dot(na,nb)
        self.assert_(scalars_equal(vc, nc, epsilon=0.0001))


    def test_complex_single_vector_vector_dot2(self):
        va = cn.array(self.complex_veca, dtype=cn.complex64)
        vb = cn.array(self.complex_vecb, dtype=cn.complex64)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_veca, dtype=np.complex64)
        nb = np.array(self.complex_vecb, dtype=np.complex64)
        nc = np.dot(na,nb)
        self.assert_(scalars_equal(vc, nc, epsilon=0.05))


    def test_complex_double_vector_vector_dot2(self):
        va = cn.array(self.complex_veca, dtype=cn.complex128)
        vb = cn.array(self.complex_vecb, dtype=cn.complex128)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_veca, dtype=np.complex128)
        nb = np.array(self.complex_vecb, dtype=np.complex128)
        nc = np.dot(na,nb)
        self.assert_(scalars_equal(vc, nc, epsilon=0.0001))


    # vector-matrix dot2 products
    def test_real_single_vector_matrix_dot2(self):
        va = cn.array(self.real_veca, dtype=cn.float32)
        vb = cn.array(self.real_matb, dtype=cn.float32)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.real_veca, dtype=np.float32)
        nb = np.array(self.real_matb, dtype=np.float32)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.05))


    def test_real_double_vector_matrix_dot2(self):
        va = cn.array(self.real_veca, dtype=cn.float64)
        vb = cn.array(self.real_matb, dtype=cn.float64)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.real_veca, dtype=np.float64)
        nb = np.array(self.real_matb, dtype=np.float64)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.0001))


    def test_complex_single_vector_matrix_dot2(self):
        va = cn.array(self.complex_veca, dtype=cn.complex64)
        vb = cn.array(self.complex_matb, dtype=cn.complex64)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_veca, dtype=np.complex64)
        nb = np.array(self.complex_matb, dtype=np.complex64)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.05))


    def test_complex_double_vector_matrix_dot2(self):
        va = cn.array(self.complex_veca, dtype=cn.complex128)
        vb = cn.array(self.complex_matb, dtype=cn.complex128)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_veca, dtype=np.complex128)
        nb = np.array(self.complex_matb, dtype=np.complex128)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.0001))


    # matrix-vector dot2 products
    def test_real_single_matrix_vector_dot2(self):
        va = cn.array(self.real_mata, dtype=cn.float32)
        vb = cn.array(self.real_vecb, dtype=cn.float32)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.real_mata, dtype=np.float32)
        nb = np.array(self.real_vecb, dtype=np.float32)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.05))


    def test_real_double_matrix_vector_dot2(self):
        va = cn.array(self.real_mata, dtype=cn.float64)
        vb = cn.array(self.real_vecb, dtype=cn.float64)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.real_mata, dtype=np.float64)
        nb = np.array(self.real_vecb, dtype=np.float64)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.0001))


    def test_complex_single_matrix_vector_dot2(self):
        va = cn.array(self.complex_mata, dtype=cn.complex64)
        vb = cn.array(self.complex_vecb, dtype=cn.complex64)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_mata, dtype=np.complex64)
        nb = np.array(self.complex_vecb, dtype=np.complex64)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.05))


    def test_complex_double_matrix_vector_dot2(self):
        va = cn.array(self.complex_mata, dtype=cn.complex128)
        vb = cn.array(self.complex_vecb, dtype=cn.complex128)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_mata, dtype=np.complex128)
        nb = np.array(self.complex_vecb, dtype=np.complex128)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.0001))


    # matrix-matrix dot2 products
    def test_real_single_matrix_matrix_dot2(self):
        va = cn.array(self.real_mata, dtype=cn.float32)
        vb = cn.array(self.real_matb, dtype=cn.float32)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.real_mata, dtype=np.float32)
        nb = np.array(self.real_matb, dtype=np.float32)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.05))


    def test_real_double_matrix_matrix_dot2(self):
        va = cn.array(self.real_mata, dtype=cn.float64)
        vb = cn.array(self.real_matb, dtype=cn.float64)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.real_mata, dtype=np.float64)
        nb = np.array(self.real_matb, dtype=np.float64)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.0001))


    def test_complex_single_matrix_matrix_dot2(self):
        va = cn.array(self.complex_mata, dtype=cn.complex64)
        vb = cn.array(self.complex_matb, dtype=cn.complex64)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_mata, dtype=np.complex64)
        nb = np.array(self.complex_matb, dtype=np.complex64)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.05))


    def test_complex_double_matrix_matrix_dot2(self):
        va = cn.array(self.complex_mata, dtype=cn.complex128)
        vb = cn.array(self.complex_matb, dtype=cn.complex128)
        vc = va.dot2(vb)
        # check we get the same answer as numpy
        na = np.array(self.complex_mata, dtype=np.complex128)
        nb = np.array(self.complex_matb, dtype=np.complex128)
        nc = np.dot(na,nb)
        self.assert_(arrays_equal(vc.toarray(), nc, epsilon=0.0001))


suite = unittest.TestLoader().loadTestsFromTestCase(TestDotProduct2)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
