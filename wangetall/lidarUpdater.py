import numpy as np
from scipy.sparse import block_diag

from cluster import Cluster
from icp import ICP
from jcbb_Cartesian import JCBB
from helper import Helper


class lidarUpdater:
    def __init__(self):
        self.cl = Cluster()
        self.jcbb = JCBB()
        self.ICP = ICP()

    def clean_up_states(self):
        pass

    def forward(self, xt, P, dt):
        """Propagates forward all tracks 
        based on current transition model"""

        F = self.calc_Fs(xt, dt)
        xt = F@xt #get index of xt's
        xt = xt.T

        Q = self.calc_Qs(xt, dt)

        P = None#a bunch of stuff, gotta index stuff.
        
        return xt, P


    def merge_tracks(self):
        pass

    def update(self, x, P, dt, data):
        xt  = x["xt"]
        xs = x["xs"]
        self.clean_up_states()
        xt, P = self.forward(xt, P, dt)
        self.associate_and_update(xt, xs, P, data)
        self.merge_tracks()

    def associate_and_update(self, xt, xs, P, data):
        #DATA ASSOCIATION *IS* THE KALMAN MEASUREMENT UPDATE!
        ### STEP 1: COURSE LEVEL ASSOCIATION
        #1. Cluster: EMST-EGBIS
        #   1a. Compute EMST over collection of points
        #   1b. Throw the EMST as input graph structure to EGBIS to compute clusters
        #   1c. Edge weights of EMST (euclidean distances between points) are taken directly
            #   as dissimilarity measure

        points_x = ranges*cos(angles)
        points_y = ranges*cos(angles)
        points = np.vstack((points_x, points_y)).T
        clusters = cl.cluster(data)
        #2. assign to static and dynamic background recursively with ICP (iterative closest point?)
        #   2a. clusters in C which contain measurements matched with boundary points in static background 
            # are associated with static background, and used to update or initialize new boundary points at fine level for static background
        #   2b. then, these clusters are removed from C, and similar process occurs recursively for each dynamic track (all other readings)
        #   2c. Clusters that remain in C at end of process are not associated with any track and each cluster will initialize
            # a tentative dynamic track.

            #ICP good bc it's after prediction step-- points will be in their predicted areas.
        
        ### STEP 2: FINE LEVEL ASSOCIATION
        #Assign to specific boundary points?
        #JCBB<-- 
        
        jcbb.run()
        pass
        



    def calc_Fs(self, xt, dt):
        F = np.zeros((6,6))
        F[0:3,0:3] = np.eye(3)
        F[0:3,3:] = dt*np.eye(3)
        F[3:, 3:] = np.eye(3)
        matrices = tuple([F for i in range(len(xt))])
        Fs = block_diag(matrices)
        return Fs

    def calc_Qs(self, xt, dt):
        V = self.calc_V()
        Q = np.zeros((6,6))
        Q[0:3,0:3] = (dt**3/3)*V
        Q[0:3,3:] = (dt**2/2)*V
        Q[3:,0:3] = (dt**2/2)*V
        Q[3:,3:] = dt*V
        matrices = tuple([Q for i in range(len(xt))])
        Qs = block_diag(matrices)
        return Qs

    def calc_V(self):
        #supposed to be a 3x3 covariance matrix for the zero mean continuous linear and angular white noise acceleration
        #supposed to be found here: https://wiki.dmdevelopment.ru/wiki/Download/Books/Digitalimageprocessing/%D0%9D%D0%BE%D0%B2%D0%B0%D1%8F%20%D0%BF%D0%BE%D0%B4%D0%B1%D0%BE%D1%80%D0%BA%D0%B0%20%D0%BA%D0%BD%D0%B8%D0%B3%20%D0%BF%D0%BE%20%D1%86%D0%B8%D1%84%D1%80%D0%BE%D0%B2%D0%BE%D0%B9%20%D0%BE%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B5%20%D1%81%D0%B8%D0%B3%D0%BD%D0%B0%D0%BB%D0%BE%D0%B2/Estimation%20with%20Applications%20to%20Tracking%20and%20Navigation/booktext@id89013302placeboie.pdf
        #but I can't find it.

        #I think it's a 3x3 diagonal...
        sigma_a = 100
        V = np.eye(3)*sigma_a

        return V
