import unittest
import test_dot
import test_dot2

# linear algebra group suite

suite = unittest.TestSuite([test_dot.suite()])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
