from scikits.learn import datasets
import gpu
import cunumpy as cn

iris = datasets.load_iris()
dev=gpu.CUDAdevice(0)

def pca_test():
    A_=cn.array(iris.data, dtype=cn.float32)
    print A_

    X_=A_.T.dot(A_)

    print X_
    print X_.toarray()

    return X_

