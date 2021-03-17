import numpy as np
from skimage.transform import estimate_transform

import matplotlib.pyplot as plt
boundaries = np.load("boundaries.npy")
boundaries_centroid = np.mean(boundaries, axis = 0)
scans = np.load("scans.npy")-boundaries_centroid
boundaries = boundaries - boundaries_centroid

# boundaries = np.arange(20).reshape((10,2))

# scans = boundaries + [1,2]

tform = estimate_transform("euclidean", boundaries, scans)
theta = tform.rotation
print(theta)
R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
print(tform.translation)

# b = np.ones((boundaries.shape[0], 3))
# b[:,0:2]= boundaries
shifted_b = tform(boundaries)
test = (boundaries.T+np.array([[tform.translation[0]], [tform.translation[1]]])).T
# plt.xlim(-0.5, 0.5)
# plt.ylim(-0.5,0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(scans[:,0], scans[:,1], c="b", marker="o", alpha =0.5, label="Scan Data")
plt.scatter(boundaries[:,0], boundaries[:,1], c="orange", marker="o", alpha = 1, label="Boundary Points")
plt.scatter(shifted_b[:,0], shifted_b[:,1], c="green", marker="o", alpha = 0.5, label="Shifted")

plt.scatter(test[:,0], test[:,1], c="purple", marker="o", alpha = 0.5, label="Test")

plt.legend()

plt.show()

