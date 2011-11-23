import unittest
import test_array
import test_linalg
import test_ml

suite = unittest.TestSuite([test_array.suite, test_linalg.suite, test_ml.suite])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)

    
