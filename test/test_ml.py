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


def suite_single():
    suite = unittest.TestSuite()
    tests = [
        'test_real_single_matrix_sum'
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
