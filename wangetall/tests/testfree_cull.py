import matplotlib.path as mplPath
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
poly = [190, 50, 500, 310]
bbPath = mplPath.Path(np.array([[poly[0], poly[1]],
                     [poly[1], poly[2]],
                     [poly[2], poly[3]],
                     [poly[3], poly[0]]]))

print(bbPath.contains_point((309, 190)))
fig, ax = plt.subplots()
patch = patches.PathPatch(bbPath, facecolor='orange', lw=2)
ax.add_patch(patch)
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
plt.show()
