import timeit
import icp
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import coarse_association
import cluster
from collections import defaultdict

dynamic_tracks_dict = defaultdict(lambda:[])
def createDictDynamicTrack(no_tracks, Q):
    for track in range(no_tracks):
        for i in range(len(Q)):
            dynamic_tracks_dict[track].append(i)

    return dynamic_tracks_dict

if __name__ == "__main__":
    start = timeit.default_timer()

    # set seed for reproducible results
    random.seed(123)

    #create dummy laser scan data
    Z = np.array((0 + np.random.random((1000,2)) * (100 - 0)))

    cl = cluster.Cluster()
    #2.: C <- CLUSTERMEASUREMENTS(Z)
    C = cl.cluster(Z)


    # create a set of points for static obejct to be the reference for ICP
    n_boundary = 10
    Q_s = np.zeros((n_boundary,2))
    Q_s[:,0] = np.array([10,20,30,40,50,60,60,60,60,60])
    Q_s[:,1] = np.array([60,60,60,60,60,50,40,30,20,10])

    # create a set of points for dynamic obejct to be the reference for ICP
    n_boundary = 8
    Q_d = np.zeros((n_boundary,2))
    Q_d[:,0] = np.array([80,85,90,90,90,85,80,80])
    Q_d[:,1] = np.array([80,80,80,75,70,70,70,75])

    dynamic_tracks_dict = createDictDynamicTrack(1, Q_d)

    ca = coarse_association.Coarse_Association(C)
    ca.run(Z, Q_s, Q_d, dynamic_tracks_dict)

    stop = timeit.default_timer()

    print('Time: ', stop - start)
