import numpy as np

z_hat = np.load("z_hat.npy")
h = np.load("h.npy")
S = np.load("S.npy")

S_inv = np.linalg.inv(S)
sub = z_hat - h
test = sub.T@S_inv@sub
print(test)
print(z_hat)
print(h)

print("Scan Pos 1:")
scanpos1 = (np.cos(z_hat[1])*z_hat[0], np.sin(z_hat[1])*z_hat[0])
print(scanpos1)

print("Tgt Pos 1:")
tgtpos1 = (np.cos(h[1])*h[0], np.sin(h[1])*h[0])
print(tgtpos1)