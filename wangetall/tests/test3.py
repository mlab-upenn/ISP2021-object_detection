import numpy as np
from skimage.transform import estimate_transform
import matplotlib.pyplot as plt

bndrs = np.load("tests/npy_files/boundaries_adjusted.npy")
scans = np.load("tests/npy_files/scans_adjusted.npy")

bndrs = bndrs - np.mean(bndrs, axis=0)
scans = scans- np.mean(scans, axis=0)
def convert_to_SE2(x):
    ret = np.hstack((x, np.ones((x.shape[0], 1))))
    return ret

tform = estimate_transform("euclidean", bndrs, scans)
print(tform)
tformed_bndrs = tform(bndrs)
tformed_bndrs2 = bndrs+np.array([-7.17736456e-19, 3.80406482e-16])
print(bndrs.shape)
H = np.array([[0.9996, -0.02469079, 0.13048156], [0.02469079, 0.99969514, 0.00534306], [0,0,1]])
H = np.array([[1, 0, 0.13048156], [0, 1, 0.00534306], [0,0,1]])
print(np.arctan2(0.02469079, 0.9996))
bndrs_expanded = convert_to_SE2(bndrs)
t = (H@bndrs_expanded.T).T
plt.scatter(scans[:,0],scans[:,1], c="green")

plt.scatter(bndrs[:,0],bndrs[:,1], c="red")
plt.scatter(tformed_bndrs[:,0],tformed_bndrs[:,1], c="orange")
plt.scatter(tformed_bndrs2[:,0],tformed_bndrs2[:,1], c="purple")

plt.show()


