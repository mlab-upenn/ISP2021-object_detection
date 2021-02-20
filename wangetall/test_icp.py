import timeit
import icp
import numpy as np
import math
import random
import matplotlib.pyplot as plt


# set seed for reproducible results
random.seed(123)

# create a set of points to be the reference for ICP
Q= np.array(random.sample(range(1000), 1000)).reshape((500,2))

points_to_be_aligned = Q[100:200]

theta = math.radians(12)
c, s = math.cos(theta), math.sin(theta)
rot = np.array([[c, -s],
                [s, c]])
P = np.dot(points_to_be_aligned, rot)
P += np.array([np.random.random_sample(), np.random.random_sample()])

icp = icp.ICP()

start = timeit.default_timer()
Z = icp.run(Q, P)
stop = timeit.default_timer()

print('Time: ', stop - start)
print(P[:3])
print(Z[:3])

# show results
# plt.plot(Q[:, 0], Q[:, 1], 'rx', label='reference points')
# plt.plot(P[:, 0], P[:, 1], 'b1', label='source points')
# plt.plot(Z[:, 0], Z[:, 1], 'g+', label='aligned points')
# plt.legend()
# plt.show()
