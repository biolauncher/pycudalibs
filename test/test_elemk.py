import unittest
import cunumpy as cn
import numpy as np
import scipy.linalg as la
import test

#
#  tests for simon's machine learning kernel library
#  these are the element/vector kernels
#
class TestElemK (unittest.TestCase):
    
    def setUp(self):
        self.real_vector = np.random.randn(2587)
        self.real_colvec = np.random.randn(80)
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

    def test_real_single_matrix_add_scalar(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        S = 27.345
        # gpu
        s = A_.add(S)
        #print s
        # cpu
        sn = A + S
        #print sn
        # XXX this precision doesn't always return true
        self.assert_(test.arrays_equal(sn, s.toarray(), 1E-03))

    def test_real_single_add_vector(self):
        V = np.array(self.real_vector, dtype=np.float32)
        V_ = cn.array(self.real_vector, dtype=cn.float32)
        # gpu
        R_ = V_.add(V_)
        # cpu
        R = V + V
        #
        self.assert_(test.arrays_equal(R, R_.toarray(), 1E-03))

    def test_real_single_matrix_add_column_vector(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        V = np.array(self.real_colvec, dtype=np.float32)
        V_ = cn.array(self.real_colvec, dtype=cn.float32)
        # gpu
        R_ = A_.add(V_)
        # cpu
        R = A + V
        #
        self.assert_(test.arrays_equal(R, R_.toarray(), 1E-03))

    def test_real_single_matrix_add_matrix(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        R_ = A_.add(A_)
        # cpu
        R = A + A
        #
        self.assert_(test.arrays_equal(R, R_.toarray(), 1E-03))

    def test_real_single_matrix_mul_scalar(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        S = 27.345
        # gpu
        s = A_.mul(S)
        #print s
        # cpu
        sn = A * S
        #print sn
        # XXX this precision doesn't always return true
        self.assert_(test.arrays_equal(sn, s.toarray(), 1E-03))

    def test_real_single_mul_vector(self):
        V = np.array(self.real_vector, dtype=np.float32)
        V_ = cn.array(self.real_vector, dtype=cn.float32)
        # gpu
        R_ = V_.mul(V_)
        # cpu
        R = V * V
        #
        self.assert_(test.arrays_equal(R, R_.toarray(), 1E-03))

    def test_real_single_matrix_mul_column_vector(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        V = np.array(self.real_colvec, dtype=np.float32)
        V_ = cn.array(self.real_colvec, dtype=cn.float32)
        # gpu
        R_ = A_.mul(V_)
        # cpu
        R = A * V
        #
        self.assert_(test.arrays_equal(R, R_.toarray(), 1E-03))

    def test_real_single_matrix_mul_matrix(self):
        A_ = cn.array(self.real_mata, dtype=cn.float32)
        A = np.array(self.real_mata, dtype=np.float32)
        # gpu
        R_ = A_.mul(A_)
        # cpu
        R = A * A
        #
        self.assert_(test.arrays_equal(R, R_.toarray(), 1E-03))


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
        'test_real_single_matrix_cproduct',
        # scalr, vector and matrix mul and add operands
        'test_real_single_matrix_add_scalar',
        'test_real_single_add_vector',
        'test_real_single_matrix_add_column_vector',
        'test_real_single_matrix_add_matrix',
        'test_real_single_matrix_mul_scalar',
        'test_real_single_mul_vector',
        'test_real_single_matrix_mul_column_vector',
        'test_real_single_matrix_mul_matrix'
        # todo math unit tests
        ]
    
    return unittest.TestSuite(map(TestElemK, tests))

def suite_double():
    suite = unittest.TestSuite()
    tests = [
        ]
    return unittest.TestSuite(map(TestElemK, tests))

suite = unittest.TestSuite([suite_single()])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
