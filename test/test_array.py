import unittest
import cunumpy as cn
import numpy as np
import math

# we will be testing cuda array creation
def arrays_equal(a, b):
    return (a==b).all()

class TestArrayFactories(unittest.TestCase):
    
    def setUp(self):
        self.real_vec = [3,4,0,11.,12,15.1,-2.,9.,14.,16.7]
        self.complex_vec = [1, 0, 2+4j, -1-math.pi*1j, 2.-3.7j, 5., 3.j, -2.3j, -math.e+math.e*1j, 7]
        self.real_mat = [self.real_vec for i in range(1,5)]
        self.complex_mat = [self.complex_vec for i in range(1,5)]

    # vectors

    def test_real_single_vector_array(self):
        # check shape, type and values
        A = cn.array(self.real_vec, dtype=cn.float32)
        self.assertEqual(A.shape, (len(self.real_vec),))
        self.assertEqual(A.dtype, cn.float32)
        self.assertEqual(A.itemsize, 4)
        self.assert_(arrays_equal(A.toarray(), np.array(self.real_vec, dtype=np.float32)))

    def test_real_double_vector_array(self):
        A = cn.array(self.real_vec, dtype=cn.float64)
        self.assertEqual(A.shape, (len(self.real_vec),))
        self.assertEqual(A.dtype, cn.float64)
        self.assertEqual(A.itemsize, 8)
        self.assert_(arrays_equal(A.toarray(), np.array(self.real_vec, dtype=np.float64)))
    
    def test_complex_single_vector_array(self):
        A = cn.array(self.complex_vec, dtype=cn.complex64)
        self.assertEqual(A.shape, (len(self.complex_vec),))
        self.assertEqual(A.dtype, cn.complex64)
        self.assertEqual(A.itemsize, 8)
        self.assert_(arrays_equal(A.toarray(), np.array(self.complex_vec, dtype=np.complex64)))

    def test_complex_double_vector_array(self):
        A = cn.array(self.complex_vec, dtype=cn.complex128)
        self.assertEqual(A.shape, (len(self.complex_vec),))
        self.assertEqual(A.dtype, cn.complex128)
        self.assertEqual(A.itemsize, 16)
        self.assert_(arrays_equal(A.toarray(), np.array(self.complex_vec, dtype=np.complex128)))

    # matrices
    def test_real_single_matrix_array(self):
        A = cn.array(self.real_mat, dtype=cn.float32)
        self.assertEqual(A.shape, (len(self.real_mat),len(self.real_vec)))
        self.assertEqual(A.dtype, cn.float32)
        self.assertEqual(A.itemsize, 4)
        self.assert_(arrays_equal(A.toarray(), np.array(self.real_mat, dtype=np.float32)))

    def test_real_double_matrix_array(self):
        A = cn.array(self.real_mat, dtype=cn.float64)
        self.assertEqual(A.shape, (len(self.real_mat),len(self.real_vec)))
        self.assertEqual(A.dtype, cn.float64)
        self.assertEqual(A.itemsize, 8)
        self.assert_(arrays_equal(A.toarray(), np.array(self.real_mat, dtype=np.float64)))
    
    def test_complex_single_matrix_array(self):
        A = cn.array(self.complex_mat, dtype=cn.complex64)
        self.assertEqual(A.shape, (len(self.complex_mat),len(self.complex_vec)))
        self.assertEqual(A.dtype, cn.complex64)
        self.assertEqual(A.itemsize, 8)
        self.assert_(arrays_equal(A.toarray(), np.array(self.complex_mat, dtype=np.complex64)))

    def test_complex_double_matrix_array(self):
        A = cn.array(self.complex_mat, dtype=cn.complex128)
        self.assertEqual(A.shape, (len(self.complex_mat),len(self.complex_vec)))
        self.assertEqual(A.dtype, cn.complex128)
        self.assertEqual(A.itemsize, 16)
        self.assert_(arrays_equal(A.toarray(), np.array(self.complex_mat, dtype=np.complex128)))

    # from miscellaneous numpy arrays - test casting
    def test_vector_cast_from_numpy_array(self):
        a = np.array([1,2,4,5])
        self.assertRaises(TypeError, cn.array, a)

    def test_matrix_cast_from_numpy_array(self):
        # default numpy is float64
        a = np.array([[1,2,4,5],[4,5,6,7],[8,9,10,11]])
        # default cuda is float32
        self.assertRaises(TypeError, cn.array, a)

    # from miscellaneous numpy arrays - test casting
    def test_vector_from_numpy_array(self):
        a = np.array([1,2,4,5], dtype=cn.float32)
        # default cuda is float32
        A = cn.array(a)
        self.assert_(arrays_equal(a, A.toarray()))

    def test_matrix_from_numpy_array(self):
        a = np.array([[1,2,4,5],[4,5,6,7],[8,9,10,11]], dtype=cn.float32)
        # default cuda is float32
        A = cn.array(a)
        self.assert_(arrays_equal(a, A.toarray()))

    # arange
    # eye
    # zeros
    # ones
    # identity



suite = unittest.TestLoader().loadTestsFromTestCase(TestArrayFactories)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
