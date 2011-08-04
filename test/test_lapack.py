import unittest
import test_svd

# lapack group suite

suite = unittest.TestSuite([test_svd.suite()
                            #,test_eigensys.suite()
                            ])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
