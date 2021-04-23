from numba import jit
import time
import numpy as np


@jit(nopython=True)
def numbized(iters, y):
    for i in range(iters):
        if y[0] < 20:
            # print(y)
            y[0]+=1
            print(y[0])
            print(y[0] < 20)
            print("calling!")
            numbized(iters, y)
    return y
class Test:
    def __init__(self):
        #prefire
        y = np.array([0])
        numbized(4, y)
    def run(self, iters, y):
        numbized(iters, y)
        return y
def main():

    test = Test()
    iters = 5
    y = np.array([0])
    y = test.run(iters, y)
    print("Final Y")
    print(y)



if __name__ == "__main__":
    main()