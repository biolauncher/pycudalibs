import unittest
import cunumpy as cn
import numpy as np
import math

# we will be testing cuda array creation
def arrays_equal(a, b):
    return (a==b).all()

class TestBugs(unittest.TestCase):
    
    def setUp(self):
        self.real_vec = [3,4,0,11.,12,15.1,-2.,9.,14.,16.7]
        self.complex_vec = [1, 0, 2+4j, -1-math.pi*1j, 2.-3.7j, 5., 3.j, -2.3j, -math.e+math.e*1j, 7]
        self.real_mat = [self.real_vec for i in range(1,5)]
        self.complex_mat = [self.complex_vec for i in range(1,5)]

    # ????
    def test_bug_00000002(self):
        A = cn.array([1,2,3,4])
        self.assertEqual(A.dtype, np.int32)

    # ????
    def test_untyped_real_vector(self):
        A = cn.array(self.real_vec)
        self.assertEqual(A.shape, (len(self.real_vec),))
        self.assertEqual(A.dtype, cn.float64)
        self.assertEqual(A.itemsize, 8)
        self.assert_(arrays_equal(A.toarray(), np.array(self.real_vec, dtype=np.float64)))
        

suite = unittest.TestLoader().loadTestsFromTestCase(TestBugs)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
