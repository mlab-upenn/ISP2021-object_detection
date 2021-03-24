import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform

scan = np.load("scan.npy")
boundary = np.load("boundaries.npy")

boundaries_centroid = np.mean(boundary, axis = 0)
boundaries_adjusted = boundary - boundaries_centroid
scans_adjusted = scan-boundaries_centroid
tform = estimate_transform("euclidean", boundaries_adjusted, scans_adjusted)
print(scans_adjusted-boundaries_adjusted)
test = tform(boundaries_adjusted)
print(tform)

# plt.scatter(selected_scan_cartesian[:,0],selected_scan_cartesian[:,1],alpha=0.5, c="red")
# for i in range(scans_adjusted.shape[0]):
#     plt.text(scans_adjusted[i,0], scans_adjusted[i,1], str(i), size = "xx-small",)


# plt.scatter(selected_bndr_pts[:,0],selected_bndr_pts[:,1],alpha=0.5, c="purple")
# for i in range(boundaries_adjusted.shape[0]):
#     plt.text(boundaries_adjusted[i,0], boundaries_adjusted[i,1], str(i), size = "xx-small")
# plt.scatter(test[:,0],test[:,1],alpha=0.5, c="green")
# plt.scatter(scans_adjusted[:,0],scans_adjusted[:,1],alpha=0.5, c="red")
# plt.scatter(boundaries_adjusted[:,0],boundaries_adjusted[:,1],alpha=0.5, c="purple")


plt.hist(scans_adjusted[:,0], alpha = 0.5, color="red")
plt.hist(boundaries_adjusted[:,0], alpha=0.5, color="purple")

plt.show()


