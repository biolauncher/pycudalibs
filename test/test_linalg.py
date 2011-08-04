import unittest
import test_dot
import test_math
import test_trans
import test_lapack

# linear algebra group suite

suite = unittest.TestSuite([test_dot.suite(),
                            test_math.suite(),
                            test_trans.suite(),
                            test_lapack.suite
                            ])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
