import unittest
import cunumpy as cn
import numpy as np
import scipy.linalg as la
import test

class TestSVD (unittest.TestCase):
    
    def setUp(self):
        D = 20.0 * np.random.random_sample((26,8)) + 1.0
        self.real_mata = D - np.mean(D, axis=0)
        # centralizer?
        self.complex_mata = np.random.randn(26,8) + 1j * np.random.randn(26,8)

    #
    # SVD 
    # TODO: make sure A_ == A after call - currently this is not the case as A_ is destroyed by LAPACK
    
    def test_real_single_matrix_svd(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        U, S, VT = [x.toarray() for x in A_.svd()]
        R = np.dot(U, np.dot(np.diag(S), VT))
        #print S
        self.assert_(test.arrays_equal(R, A, 1e-03))


    def test_real_double_matrix_svd(self):
        A_ = cn.array(self.real_mata, dtype=cn.float64)
        A = np.array(self.real_mata, dtype=np.float64)
        # gpu
        U, S, VT = [x.toarray() for x in A_.svd()]
        R = np.dot(U, np.dot(np.diag(S), VT))
        #print S
        self.assert_(test.arrays_equal(R, A, 1e-03))
       

    def test_complex_single_matrix_svd(self):
        A_ = cn.array(self.complex_mata, dtype=cn.complex64)
        A = np.array(self.complex_mata, dtype=np.complex64)
        # gpu
        U, S, VT = [x.toarray() for x in A_.svd()]
        R = np.dot(U, np.dot(np.diag(S), VT))
        self.assert_(test.arrays_equal(R, A, 1e-03))
       
       
    def test_complex_double_matrix_svd(self):
        A_ = cn.array(self.complex_mata, dtype=cn.complex128)
        A = np.array(self.complex_mata, dtype=np.complex128)
        # gpu
        U, S, VT = [x.toarray() for x in A_.svd()]
        R = np.dot(U, np.dot(np.diag(S), VT))
        #print S
        self.assert_(test.arrays_equal(R, A, 1e-03))


def suite_single():
    suite = unittest.TestSuite()
    tests = ['test_real_single_matrix_svd'
             ,'test_complex_single_matrix_svd'
             ]

    return unittest.TestSuite(map(TestSVD, tests))

def suite_double():
    suite = unittest.TestSuite()
    tests = ['test_real_double_matrix_svd'
             ,'test_complex_double_matrix_svd'
             ]
    return unittest.TestSuite(map(TestSVD, tests))

def suite():
    return unittest.TestSuite([suite_single()])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
