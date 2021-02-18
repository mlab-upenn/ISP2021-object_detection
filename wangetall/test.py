# import numpy as np
# from scipy.sparse import block_diag


# G_indiv=np.array([[2,0],[0,2]])
# G_matrices = tuple([G_indiv for i in range(9)])
# Gs = block_diag(G_matrices)
# print(Gs.shape)

# v = np.zeros((18,2)) 
# v[:] = np.ones((2,2))
# print(v)
# # v[1::2] = np.array([0,0])
# print("new v")
# print(v)
# b=v.T@Gs
# print(b.shape)
# print(b.T[::2])
# # print(b[::2])

def test(P):
    def test2():
        print(P)
    test2()

test(2)