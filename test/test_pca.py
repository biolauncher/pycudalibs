from scikits.learn import datasets
import gpu
import cunumpy as cn

iris = datasets.load_iris()
dev=gpu.CUDAdevice(0)
A_=cn.array(iris.data, dtype=cn.float32)

def pca_test():
    # TODO centralizer kernel
    X_ = A_.add(A_.csum().mul(-1))
    Y_= X_.T.dot(X_).eigensystem(pure=False)    # no need to preserve intermediate matrix
    return Y_

