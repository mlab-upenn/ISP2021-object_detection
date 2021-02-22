import numpy as np
from scipy import stats

indiv_compat = np.load("indiv_compat.npy")
print(indiv_compat[28])


#scan 28--> boundary 1: -0.00455899, -0.05761841; JNIS 3.34e-02

#scan73 --> boundary 1: -4.28608035e-01, 1.39731475e-03; JNIS: 1.837

#ok... so it seems that the problem here may be in the fact that we're subtracting in polar coordinates...
#as expected, the angle diff between the scan 28 - boundary 1 matches are larger than
#the diff between scan 73-boundary 1 diff. 