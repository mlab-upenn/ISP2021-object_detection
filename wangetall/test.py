import numpy as np
from skimage.transform import estimate_transform
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

pts = np.load("pts.npy")
refs = np.load("ref.npy")

C = cdist(pts, refs)
print(refs.shape)
print(pts.shape)
print(C.shape)
_, assigment = linear_sum_assignment(C)

if pts.shape[0] < refs.shape[0]:
    N = pts.shape[0]
else:
    N = refs.shape[0]


validIdxs = [i for i in range(N) if C[i, assigment[i]]<10]
pairs = np.zeros((len(validIdxs),2, 2))
pairs[:,:,0] = pts[validIdxs]
pairs[:,:,1] = refs[assigment[validIdxs]]
print(assigment.shape)

plt.plot(pts[:,0], pts[:,1],'bo', markersize = 10)
plt.plot(refs[:,0], refs[:,1],'rs',  markersize = 7)
for p in range(refs.shape[0]):
    plt.plot([pts[p,0], refs[assigment[p],0]], [pts[p,1], refs[assigment[p],1]], 'k')
# plt.xlim(-1.1,1.1)
# plt.ylim(-1.1,1.1)
plt.axes().set_aspect('equal')
plt.show()

# x = [[(0,0), (0.1,0.1)],[(2,2), (2.1, 2.1)], [(3,3), (3.1, 3.1)]]
# t = np.array(x)
# print(t)
# print(t[:,0,:])
# print(t.shape)

# tform = estimate_transform("euclidean", t[:,0,:], t[:,1,:])
# dx, dy = tform.translation
# print(dx)
