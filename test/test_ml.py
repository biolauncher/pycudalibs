import unittest
import test_elemk
import test_svd

#
#  tests for simon's machine learning kernel library 
#
suite = unittest.TestSuite([test_elemk.suite, test_svd.suite])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
