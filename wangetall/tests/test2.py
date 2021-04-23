import numpy as np
import time
from scipy import linalg
from numba import njit
from scipy.linalg import block_diag
x = np.ones((1000))
M = np.diag(np.arange(1,1001))
# print(M)
i = np.random.choice(1000, size=1000, replace=False)

i.sort()
print(i)
start = time.time()
x_sub = np.ones((len(i)))
M_sub = M[i[:,None],i]
S_m = np.linalg.inv(M_sub)
print(x_sub.T@S_m@x_sub)
end = time.time()
print("Time1 {}".format(end-start))

L = np.linalg.cholesky(M)

start = time.time()
L_sub = L[i[:,None],i]
y = np.linalg.solve(L_sub, x_sub)
c = np.linalg.norm(y)**2
print(c)
end = time.time()
print("Time2 {}".format(end-start))
# S = np.random.randint(0, 100, 25).reshape((5,5))
# S_i = np.linalg.inv(S)
# print(S_i)
# i = np.array([0,2, 3])
# S2 = S[i[:, None], i]
# S2_i = np.linalg.inv(S2)
# print(S2_i)
# S1_i = S_i[i[:,None],i]
# print(S1_i)


def chol(x, S):
    L = np.linalg.cholesky(S)
    y = linalg.solve_triangular(L, x)
    c = np.linalg.norm(y)**2
    return c



# # start = time.time()
# # zz , _ = linalg.lapack.dpotrf(M, False, False)
# # inv_M , info = linalg.lapack.dpotri(zz)
# # inv_M = np.triu(inv_M) + np.triu(inv_M, k=1).T
# # end = time.time()
# # print("time1 {}".format(end-start))

# start2 = time.time()
# b = np.linalg.solve(M, np.eye(M.shape[0]))
# end2 = time.time()
# print("time2 {}".format(end2-start2))


# # @njit(fastmath=False, cache = True)
# # def matrix_inverse(M):
# #     return np.linalg.inv(M)

# start3 = time.time()
# L = np.linalg.cholesky(M)
# print(L.shape)
# y = linalg.solve_triangular(L, x)
# c = np.linalg.norm(y)**2
# print(c)
# end3 = time.time()
# print("Time {}".format(end3-start3))

# # for i in range(t):
# #     G[i*2:i*2+2] = np.eye(2).T
