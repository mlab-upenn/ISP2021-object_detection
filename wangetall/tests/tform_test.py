import numpy as np
from skimage.transform import estimate_transform
import matplotlib.pyplot as plt

bndrs = np.load("tests/npy_files/boundaries_adjusted.npy")
scans = np.load("tests/npy_files/scans_adjusted.npy")

bndrs = bndrs - np.mean(bndrs, axis=0)
scans = scans- np.mean(bndrs, axis=0)

def convert_to_SE2(x):
    ret = np.hstack((x, np.ones((x.shape[0], 1))))
    return ret

bndrs_interped_x = np.linspace(np.min(bndrs[:,0]), np.max(bndrs[:,0]), num = len(bndrs[:,0])*10)
bndrs_interped_y = np.interp(bndrs_interped_x, bndrs[:,0], bndrs[:,1])
bndrs_interped = np.vstack((bndrs_interped_x, bndrs_interped_y)).T

scans_interped_x = np.linspace(np.min(scans[:,0]), np.max(scans[:,0]), num = len(scans[:,0])*10)
scans_interped_y = np.interp(scans_interped_x, scans[:,0], scans[:,1])
scans_interped = np.vstack((scans_interped_x, scans_interped_y)).T

tform = estimate_transform("euclidean", bndrs_interped, scans_interped)
tformed_bndrs = tform(bndrs)
# H = np.array([[0.9996, -0.02469079, 0.13048156], [0.02469079, 0.99969514, 0.00534306], [0,0,1]])
# H = np.array([[1, 0, 0.13048156], [0, 1, 0.00534306], [0,0,1]])
bndrs_expanded = convert_to_SE2(bndrs)
tformed_bndrs2 = (tform.params@bndrs_expanded.T).T[:,0:2]
print(tform)
plt.scatter(scans[:,0],scans[:,1], c="green", label="Scans")


plt.scatter(bndrs[:,0],bndrs[:,1], c="red", label="Boundaries")
plt.scatter(tformed_bndrs[:,0],tformed_bndrs[:,1], c="orange", label="Transformed Boundaries")
# plt.scatter(tformed_bndrs2[:,0],tformed_bndrs2[:,1], c="purple")
plt.legend()
plt.show()
breakpoint()


