# test the implications of library shutdown - if this is a problem
# we may need just to change the call to shutdown in close to free all
# CUDA memory instead.

import gpu
import cunumpy as cn

def with_device():
    with gpu.device(0) as cuda0:
        A = cn.ones((1024,8), dtype=cn.float32)
        print A

# this seems to crash ipython!
with_device()

