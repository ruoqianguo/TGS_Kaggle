# Cython version rlenc
# Source https://www.kaggle.com/bguberfain/unet-with-depth
import numpy as np
cimport numpy as np

cdef extern from "rlen_core.h":
    int rlen(int * input, int len, int * ouput)

def RLenc(np.ndarray[int, ndim=1, mode="c"] input not None):
    cdef H = input.shape[0]
    cdef W = input.shape[1]
    cdef np.ndarray[int, ndim=2, mode="c"] output = np.zeros((H * W, 2), dtype=np.int32)
    cdef int cnt = rlen(<int *> np.PyArray_DATA(input), input.shape[0], <int *> np.PyArray_DATA(output))
    output = output[:cnt].reshape(-1, 2)
    return output
