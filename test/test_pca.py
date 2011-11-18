from scikits.learn import datasets
import gpu
import cunumpy as cn

iris = datasets.load_iris()
dev=gpu.CUDAdevice(0)
A_=cn.array(iris.data, dtype=cn.float32)


def centralise(A):
    return A.add(A.csum().mul(-1./A.shape[0]))

def pca_test(A):
    X = centralise(A)
    return X.T.dot(X).eigensystem(pure=False)    # no need to preserve intermediate matrix


