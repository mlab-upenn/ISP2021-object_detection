import numpy as np
from scipy.linalg import block_diag

x = np.ones((10, 2, 2))
# Rs = block_diag(*R_matrices)

R = block_diag(*x)
print(R)