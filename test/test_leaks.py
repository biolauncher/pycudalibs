import unittest
import cunumpy as cn
import numpy as np
import math
import test

class TestLeaks (unittest.TestCase):

    
    def setUp(self):
        self.n = 10000
        self.a = np.ones((4096,4096))
        
    def test_memory_churn(self):
        for x in xrange(self.n):
            b = cn.array(self.a, dtype=cn.float32)
            #b = a.sqrt()
            c = b.dot(b)
            del b, c
        self._assert(True)

def suite():
    suite = unittest.TestSuite()
    tests = ['test_memory_churn']

    return unittest.TestSuite(map(TestLeaks, tests))


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
