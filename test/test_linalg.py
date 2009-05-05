import unittest
import test_dot
import test_math

# linear algebra group suite

suite = unittest.TestSuite([test_dot.suite(), test_math.suite()])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
