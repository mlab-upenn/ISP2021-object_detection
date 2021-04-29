import numpy as np
from numba import jit
import time


def solve(A, b):
    return np.linalg.solve(A, b)

@jit(nopython=True)
def numba_solve(A,b):
    return np.linalg.solve(A,b)



if __name__ == "__main__":
    A = np.eye(250)
    b = np.ones(250)
    numba_solve(A,b)
    st = time.time()
    solve(A,b)
    et = time.time()
    print("Non-numbatime {}".format(et-st))
    st = time.time()
    numba_solve(A,b)
    et = time.time()
    print("Numba time {}".format(et-st))