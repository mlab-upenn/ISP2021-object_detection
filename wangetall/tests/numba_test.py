from numba import jit
import time
import numpy as np


@jit(nopython=True)
def numbized(iters, boost):
    x = 0
    for i in range(iters):
        print(y)
        x+=boost
def non_numbized(iters, boost):
    x = 0
    for i in range(iters):
        x+=boost
class Test:
    def __init__(self):
        #prefire
        y = np.array([3])
        numbized(10099, 1)
        time.sleep(9)
    def run(self, iters, boost):
        numbized(iters, boost)
def main():

    test = Test()
    iters = 10000
    boost = 10.33
    st = time.time()
    non_numbized(iters, boost)
    et = time.time()
    st_n = time.time()
    test.run(iters, boost)
    et_n = time.time()

    print("Numba time {}, regular time {}".format(et_n-st_n, et-st))


if __name__ == "__main__":
    main()