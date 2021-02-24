import numpy as np
from scipy import stats





class init_and_merg:
    def __init__(self):
        self.alpha = 1-0.95
        pass

    def run(self, x_tentative, P, xp, xt):
        self.x_tentative = x_tentative #xt is a 2D matrix where each row corresponds with a
                    #new tentative track T. Columns are [gamma, delta, phi, gamma_dot, delta_dot, phi_dot]
        self.P = P
        self.xp = xp

        self.static_check()
        self.xt = xt

    def static_check(self):
        static_check = np.zeros((self.x_tentative.shape[0]))
        chi2 = stats.chi2.ppf(self.alpha, df=3)

        h = self.x_tentative[:, 3:6] #fictious measurement
        h = np.reshape(h, (h.shape[0],1,-1)) #reshape into 3D matrix, where vectors of 3 are stacked on top of each other
        #optimization: move creation of H and z_hat into init to avoid calling every time.
        H= np.zeros((h.shape[0], 3, 6))
        H[:] = np.hstack((np.eye(3), np.ones((3,3))))
        S = self.P #do I need to reshape P to make S into a 3D matrix?
        z_hat = np.zeros((h.shape[0], 3, 1))
        z_hat[:] = np.zeros((3,1))

        a = z_hat - h
        b = np.linalg.inv(S)
        val = np.einsum('ki,kij,kj->k', a, b, a)

        static_check[np.where(val <= chi2)] = 1 #1 indicates that the track has been flagged by
                                                #the static check, and should be merged into the static backgorund.

        #merge. How?
    

    def dynamic_check(self):
        for track in self.xt:
            
            h2 = 



#want only *dynamic* objects to be initialized as new tracks. 
# Static objects should be merged with the background.
# 
# New tracks first marked as tentative. Then, it's marked as "mature" 
# only if it's observed continuously for a fixed number of frames. Otherwise, it's dropped.
# Then, it is tested first against the background, and then against each of the dynamic tracks.
# If it's identified for merging, it is merged-- otherwise, if all merging tests fail,
# It is declared "established" and added to the set of dynamic tracks.

####ALL OF THIS OCCURS AFTER COARSE ASSOCIATION, NOT FINE ASSOCIATION!

#How do we track whether a track is viewed continuously??
#-After ICP identifies unassociated cluster, keep it in memory. Then, add
# it to the ICP boundary points, and see if a new cluster is matched to it.
# If so, keep it in memory as an ICP boundary point, and see if a new cluster gets matched.
# Repeat N times, where N is the maturity constant.

#### 1. The static test.
# Define fictious measurement model h = [gamma_dot, delta_dot, phi_dot]. 
# Consider measurement z_hat = 0 under a noise free condition R = 0.
# The idea is that given an observer who perfectly observes the internal dynamics of the 
# object, what's the probability that they'll say that they are 0?


#Since R = 0, S = P, where P is Pv_tv_t -- the submatrix of P corresponding to the 
#velocity of track t.

#Do validation test on measurement z_hat = 0. See if it falls within the validation gate
# (chi2 test) 