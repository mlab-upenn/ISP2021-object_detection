import numpy as np

z_hat = np.load("zhat.npy")
h = np.load("h.npy")
S = np.load("S.npy")*50

S_inv = np.linalg.inv(S)
sub = z_hat - h
test = sub.T@S_inv@sub
print(test)