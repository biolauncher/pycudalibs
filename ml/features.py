"""
feature space tools
data vectors are assumed to be in columns of matrices (cunumpy arrays)
seb@modelsciences.com
"""

# no need to import specific symbols as we simply use methods on array objects
# these are expected to be cunumpy arrays N.B. if you use numpy arrays YMMV

def centralise(A):
    """
    Return a matrix with the data (column) vectors normalised to zero
    sum.
    """
    return A.add(A.csum().mul(-1./A.shape[0]))

def centralize(A):
    """
    American english for centralise.
    """
    return centralise(A)




def singular_values(A):
    """
    Compute singular values only by taking square roots of
    eigenvalues of (A.T * A), the column vectors in A are first
    centralised and the singular values are returned without
    sorting so that the principal components can be identified
    w.r.t the data matrix A.
    """
    X = centralise(A)
    return X.T.dot(X).eigensystem(pure=False)[0].sqrt()

def svd(A):
    """
    Compute the SVD decomposition of A after centralising its data
    vectors.
    """
    return centralise(A).svd()
