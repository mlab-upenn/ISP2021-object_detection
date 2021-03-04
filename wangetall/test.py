import sys
import time
import gc
import os, psutil
import numpy as np
import copy
class Test:
    def __init__(self, idx):
        self.id = idx
        self.nums = np.arange(10000000)
    def del_nums(self):
        del self.nums
    def __del__(self):
        print("Destructor called!")
state = {}
def add_to_state():
    t = Test(0)
    state[0] = t


if __name__ == "__main__":
    add_to_state()
    state.pop(0)
    b = copy.deepcopy(state)
    del state
    gc.collect()
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)  # in bytes 

    time.sleep(5)


