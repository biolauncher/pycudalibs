from scikits.learn import datasets
import gpu
import cunumpy as cn

iris = datasets.load_iris()
dev=gpu.CUDAdevice(0)

def pca_test():
    A_=cn.array(iris.data, dtype=cn.float32)
    # TODO centralizer kernel
    X_=A_.T.dot(A_).eigensystem(pure=False)    # no need to preserve intermediate matrix
    return X_

