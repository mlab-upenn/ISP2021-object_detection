import numpy as np
import matplotlib.pyplot as plt

gt = np.load("true_speed.npy")
output = np.load("Tracked_obj_speed.npy")

plt.plot(output[:550,2], label="Output")
plt.plot(gt[:550], label="Ground Truth")
loss = np.sqrt(np.abs(np.sum(output[:550,2]**2- gt[:550]**2)))
print(loss)
plt.xlabel("Iteration")
plt.ylabel("Velocity (m/s)")
plt.grid(which="major")
plt.legend()
plt.show()