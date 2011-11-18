import unittest
import test_svd
import test_eigensystem

# lapack group suite

suite = unittest.TestSuite([test_svd.suite()
                            ,test_eigensystem.suite()
                            ])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
