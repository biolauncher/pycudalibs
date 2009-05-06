import unittest
import test_array
import test_linalg

suite = unittest.TestSuite([test_array.suite, test_linalg.suite])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)

    
