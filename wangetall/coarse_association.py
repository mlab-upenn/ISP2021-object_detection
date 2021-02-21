import icp
import cluster
import random
import numpy as np
import matplotlib.pyplot as plt

def associateAndUpdateWithStatic(C, Z, Q):
    icp_obj = icp.ICP()
    static_C = {}
    for key in C.keys():
        P = Z[C[key]]
        static = icp_obj.run(Q, P)
        if static:
            static_C[key] = C[key]
    return static_C

if __name__ == "__main__":
    cl = cluster.Cluster()

    # set seed for reproducible results
    random.seed(1234)

    # create a set of points to be the reference for ICP
    Z = np.array(random.sample(range(200), 200)).reshape((100,2))
    C = cl.cluster(Z)

    n_boundary = 10
    Q = np.zeros((n_boundary,2))
    Q[:,0] = np.array([10,20,30,40,50,60,60,60,60,60])
    Q[:,1] = np.array([60,60,60,60,60,50,40,30,20,10])

    static_C = associateAndUpdateWithStatic(C, Z, Q)

    for key in C.keys():
        P = Z[C[key]]
        plt.scatter(P[:,0], P[:,1])
    plt.show()

    for key in static_C.keys():
        static_P = Z[static_C[key]]
        data = plt.scatter(Z[:,0],Z[:,1])
        static_cloud = plt.scatter(static_P[:,0],static_P[:,1])
        border = plt.scatter(Q[:,0],Q[:,1], marker='x')
        plt.legend((data, static_cloud, border),
                    ("laser scan data", "static cloud", "boundary points"))
    plt.show()
