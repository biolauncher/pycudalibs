import unittest
import cunumpy as cn
import numpy as np
import scipy.linalg as la
import test

class TestEigensystem (unittest.TestCase):
    
    def setUp(self):
        self.real_mata = np.random.randn(10,10)
        self.complex_mata = np.random.randn(10,10) + 1j * np.random.randn(10,10)

    #
    # SVD
    
    # make sure A_ == A after call
    def test_fn_purity(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        RU, IU = [x.toarray() for x in A_.eigensystem()]
        self.assert_(test.arrays_equal(A, A_.toarray()))
        
    
    def test_real_single_matrix_eigensystem(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        LV, RU, IU, RV = [x.toarray() for x in A_.eigensystem(left_vectors=True, right_vectors=True)]
        E = RU + 1j * IU # glue egienvalues into complex array
        #
        NE, NLV, NRV = la.eig(A, left=True, right=True)
        self.assert_(test.arrays_equal(E, NE, 1e-03))


    def test_real_double_matrix_eigensystem(self):
        A_ = cn.array(self.real_mata, dtype=cn.float64)
        A = np.array(self.real_mata, dtype=np.float64)
        # gpu
        LV, E, RV = [x.toarray() for x in A_.eigensystem(left_vectors=True, right_vectors=True)]
        # cpu
        NE, NLV, NRV = la.eig(A, left=True, right=True)
        self.assert_(test.arrays_equal(E, NE, 1e-03))
       

    def test_complex_single_matrix_eigensystem(self):
        A_ = cn.array(self.complex_mata, dtype=cn.complex64)
        A = np.array(self.complex_mata, dtype=np.complex64)
        # gpu
        LV, E, RV = [x.toarray() for x in A_.eigensystem(left_vectors=True, right_vectors=True)]
        # cpu
        NE, NLV, NRV = la.eig(A, left=True, right=True)
        self.assert_(test.arrays_equal(E, NE, 1e-03))
       
       
    def test_complex_double_matrix_eigensystem(self):
        A_ = cn.array(self.complex_mata, dtype=cn.complex128)
        A = np.array(self.complex_mata, dtype=np.complex128)
        # gpu
        LV, E, RV = [x.toarray() for x in A_.eigensystem(left_vectors=True, right_vectors=True)]
        # cpu
        NE, NLV, NRV = la.eig(A, left=True, right=True)
        self.assert_(test.arrays_equal(E, NE, 1e-03))
       


def suite_single():
    suite = unittest.TestSuite()
    tests = [
        'test_fn_purity',
        'test_real_single_matrix_eigensystem',
        'test_complex_single_matrix_eigensystem'
        ]

    return unittest.TestSuite(map(TestEigensystem, tests))

def suite_double():
    suite = unittest.TestSuite()
    tests = [
        'test_real_double_matrix_eigensystem',
        'test_complex_double_matrix_eigensystem'
        ]
    return unittest.TestSuite(map(TestEigensystem, tests))

def suite():
    return unittest.TestSuite([suite_single()])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
