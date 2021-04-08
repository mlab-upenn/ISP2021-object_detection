import numpy as np
from scipy.linalg import block_diag


G  = np.zeros((10, 2, 2))
G[:] = np.eye(2).T
print(G.shape)

G2 = np.tile(np.eye(2).T, (10, 1))
print(G2.shape)
G3 = G.reshape((G.shape[0]*2, 2))
print(G3)

# x = np.ones((10, 2, 2))
# # Rs = block_diag(*R_matrices)

# R = block_diag(*x)
# print(R)