import numpy as np

x = np.arange(10)
print(x)
y = np.zeros((x.shape[0]*2))
y[::2] = x
y[1::2]=x+1
print(y)