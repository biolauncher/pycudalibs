import unittest
import cunumpy as cn
import numpy as np
import scipy.linalg as lalg
import math

# we will be testing linear algebra on CUDA we we need an approximation
def arrays_equal(a, b, epsilon=0.000001):
    return (abs(a-b) < epsilon).all()

def scalars_equal(a, b, epsilon=0.00004):
    #print a, b, abs(a-b)
    return abs(a-b) < epsilon

class TestMath(unittest.TestCase):
    
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

    def test_scale_real_vector(self):
        A = cn.array(self.real_veca, dtype=cn.float32)
        B = A.multiply(math.pi)
        # numpy
        a = np.array(self.real_veca, dtype=np.float32)
        b = np.multiply(a, math.pi)
        self.assert_(arrays_equal(B.toarray(), b)) 

    def test_scalez_complex_vector(self):
        A = cn.array(self.complex_veca, dtype=cn.complex64)
        B = A.multiply(math.pi+1.37j)
        # numpy
        a = np.array(self.complex_veca, dtype=np.complex64)
        b = np.multiply(a, math.pi+1.37j)
        self.assert_(arrays_equal(B.toarray(), b)) 

    def test_scale_complex_vector(self):
        A = cn.array(self.complex_veca, dtype=cn.complex64)
        B = A.multiply(math.pi)
        # numpy
        a = np.array(self.complex_veca, dtype=np.complex64)
        b = np.multiply(a, math.pi)
        self.assert_(arrays_equal(B.toarray(), b)) 

    def test_scalez_real_vector(self):
        A = cn.array(self.real_veca, dtype=cn.float32)
        # exception cannot yet do this
        self.assertRaises(ValueError, A.multiply, math.pi+1.j)


    def test_scale_real_matrix(self):
        A = cn.array(self.real_mata, dtype=cn.float32)
        B = A.multiply(math.pi)
        # numpy
        a = np.array(self.real_mata, dtype=np.float32)
        b = np.multiply(a, math.pi)
        self.assert_(arrays_equal(B.toarray(), b)) 

    def test_scalez_complex_matrix(self):
        A = cn.array(self.complex_mata, dtype=cn.complex64)
        B = A.multiply(math.pi+1.37j)
        # numpy
        a = np.array(self.complex_mata, dtype=np.complex64)
        b = np.multiply(a, math.pi+1.37j)
        self.assert_(arrays_equal(B.toarray(), b)) 

    def test_scale_complex_matrix(self):
        A = cn.array(self.complex_mata, dtype=cn.complex64)
        B = A.multiply(math.pi)
        # numpy
        a = np.array(self.complex_mata, dtype=np.complex64)
        b = np.multiply(a, math.pi)
        self.assert_(arrays_equal(B.toarray(), b)) 

    def test_scalez_real_matrix(self):
        A = cn.array(self.real_mata, dtype=cn.float32)
        # exception cannot yet do this
        self.assertRaises(ValueError, A.multiply, math.pi+1.j)

    # norms

    def test_norm_real_vector(self):
        A = cn.array(self.real_veca, dtype=cn.float32)
        # numpy
        a = np.array(self.real_veca, dtype=np.float32)
        self.assert_(scalars_equal(lalg.norm(a), A.norm()))

    def test_norm_real_matrix(self):
        A = cn.array(self.real_mata, dtype=cn.float32)
        # numpy
        a = np.array(self.real_mata, dtype=np.float32)
        self.assert_(scalars_equal(lalg.norm(a), A.norm()))

    def test_norm_complex_vector(self):
        A = cn.array(self.complex_veca, dtype=cn.complex64)
        # numpy
        a = np.array(self.complex_veca, dtype=np.complex64)
        self.assert_(scalars_equal(lalg.norm(a), A.norm()))

    def test_norm_complex_matrix(self):
        A = cn.array(self.complex_mata, dtype=cn.complex64)
        # numpy
        a = np.array(self.complex_mata, dtype=np.complex64)
        self.assert_(scalars_equal(lalg.norm(a), A.norm()))

    # asum - asum is not much use anyway not even a proper L1 norm for complex

    def test_asum_real_vector(self):
        A = cn.array(self.real_veca, dtype=cn.float32)
        # numpy
        a = np.array(self.real_veca, dtype=np.float32)
        self.assert_(scalars_equal(np.sum(np.abs(a)), A.asum()))

    def test_asum_real_matrix(self):
        A = cn.array(self.real_mata, dtype=cn.float32)
        # numpy
        a = np.array(self.real_mata, dtype=np.float32)
        self.assert_(scalars_equal(np.sum(np.abs(a)), A.asum()))

    def test_asum_complex_vector(self):
        A = cn.array(self.complex_veca, dtype=cn.complex64)
        # numpy
        a = np.array(self.complex_veca, dtype=np.complex64)
        self.assert_(scalars_equal(np.sum(np.abs(a)), A.asum()))

    def test_asum_complex_matrix(self):
        A = cn.array(self.complex_mata, dtype=cn.complex64)
        # numpy
        a = np.array(self.complex_mata, dtype=np.complex64)
        self.assert_(scalars_equal(np.sum(np.abs(a)), A.asum()))
    

def suite_single():
    suite = unittest.TestSuite()
    tests = ['test_scale_real_vector',
             'test_scalez_complex_vector',
             'test_scale_complex_vector',
             'test_scalez_real_vector',
             'test_scale_real_matrix',
             'test_scalez_complex_matrix',
             'test_scale_complex_matrix',
             'test_scalez_real_matrix',
             'test_norm_real_vector',
             'test_norm_complex_vector',
             'test_norm_real_matrix',
             'test_norm_complex_matrix',
             'test_asum_real_vector',
             #'test_asum_complex_vector',
             'test_asum_real_matrix'
             #'test_asum_complex_matrix'
             ]

    return unittest.TestSuite(map(TestMath, tests))

def suite_double():
    suite = unittest.TestSuite()
    tests = []
    return unittest.TestSuite(map(TestMath, tests))

def suite():
    return unittest.TestSuite(suite_single())

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
