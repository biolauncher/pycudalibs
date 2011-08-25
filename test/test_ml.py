import unittest
import cunumpy as cn
import numpy as np
import scipy.linalg as la
import test

#
#  tests for simon's machine learning kernel library 
#
class TestML (unittest.TestCase):
    
    def setUp(self):
        self.real_vector = np.random.randn(2578)
        self.real_mata = np.random.randn(2602, 80)
        self.complex_vector = np.random.randn(2587) + 1j * np.random.randn(2587)
        self.complex_mata = np.random.randn(2602,80) + 1j * np.random.randn(2602,80)

    #
    # cudaml tests
    #
    
    def test_real_single_matrix_sum(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        s = A_.sum()
        # cpu
        sn = np.sum(A)
        # XXX this precision doesn't always return true
        self.assert_(test.scalars_equal(sn, s, 1E-04))


    def test_real_single_matrix_max(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        s = A_.max()
        # cpu
        sn = np.max(A)
        # XXX this precision doesn't always return true
        self.assert_(test.scalars_equal(sn, s, 1E-04))

    def test_real_single_matrix_min(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        s = A_.min()
        # cpu
        sn = np.min(A)
        # XXX this precision doesn't always return true
        self.assert_(test.scalars_equal(sn, s, 1E-04))

    def test_real_single_matrix_product(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        s = A_.product()
        # cpu
        sn = np.product(A)
        # XXX this precision doesn't always return true
        self.assert_(test.scalars_equal(sn, s, 1E-04))

    def test_real_single_matrix_csum(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        s = A_.csum().toarray()
        #print s
        # cpu
        sn = np.sum(A, axis=0)
        #print sn
        # XXX this precision doesn't always return true
        self.assert_(test.arrays_equal(sn, s, 1E-03))

    def test_real_single_matrix_cmax(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        s = A_.cmax().toarray()
        #print s
        # cpu
        sn = np.max(A, axis=0)
        #print sn
        # XXX this precision doesn't always return true
        self.assert_(test.arrays_equal(sn, s, 1E-03))

    def test_real_single_matrix_cmin(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        s = A_.cmin().toarray()
        #print s
        # cpu
        sn = np.min(A, axis=0)
        #print sn
        # XXX this precision doesn't always return true
        self.assert_(test.arrays_equal(sn, s, 1E-03))

    def test_real_single_matrix_cproduct(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        s = A_.cproduct().toarray()
        #print s
        # cpu
        sn = np.product(A, axis=0)
        #print sn
        # XXX this precision doesn't always return true
        self.assert_(test.arrays_equal(sn, s, 1E-03))
        
def suite_single():
    suite = unittest.TestSuite()
    tests = [
        'test_real_single_matrix_sum',
        'test_real_single_matrix_max',
        'test_real_single_matrix_min',
        'test_real_single_matrix_product',
        'test_real_single_matrix_csum',
        'test_real_single_matrix_cmax',
        'test_real_single_matrix_cmin',
        'test_real_single_matrix_cproduct'
        ]
    
    return unittest.TestSuite(map(TestML, tests))

def suite_double():
    suite = unittest.TestSuite()
    tests = [
        ]
    return unittest.TestSuite(map(TestML, tests))

def suite():
    return unittest.TestSuite([suite_single()])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
