# module of support routines for testing
import numpy as np

# we will be testing linear algebra on CUDA we we need an approximation
def arrays_equal(a, b, epsilon=0.000001):
    return np.allclose(a,b)
    #return (abs(a-b) < epsilon).all()

def close(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= (atol + rtol * abs(b))

def scalars_equal(a, b, epsilon=0.00004):
    #print a, b, abs(a-b)
    return close(a, b)
    #return abs(a-b) < epsilon
