import numpy as np
import rospy
from scipy.sparse import block_diag
from cluster import Cluster
from icp import ICP
class Helper:
    @staticmethod
    def compute_rot_matrix(angle):
        R = np.array([[np.cos(angle), -np.sin(angle)], 
                        [np.sin(angle), np.cos(angle)]])
        return R 

    @staticmethod
    def quat_to_rot(quat):
        R = np.zeros((3,3))
        R[0,0] = 2*(quat[0]**2+quat[1]**2)-1
        R[1,0] = 2*(quat[1]*quat[2]+quat[0]*quat[3])
        R[2,0] = 2*(quat[1]*quat[3]+quat[0]*quat[2])

        R[0,1] = 2*(quat[1]*quat[2]-quat[0]*quat[3])
        R[1,1] = 2*(quat[0]**2+quat[2]**2)-1
        R[2,1] = 2*(quat[2]*quat[3]+quat[0]*quat[1])
        R[0,2] = 2*quat(quat[1]*quat[3]+quat[0]*quat[2])
        R[1,2] = 2*(quat[2]*quat[3]-quat[0]*quat[1])
        R[2,2] = 2*(quat[0]**2+quat[3]**2)-1


        #can probs ignore the cells on the bottom right edge bc 2D
        return R

class lidarUpdater:
    def __init__(self):
        pass


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
        self.clean_up_states()
        xt, P = self.forward(xt, P, dt)
        self.associate_and_update(xt, P, data)
        self.merge_tracks()

    def associate_and_update(self, xt, P, data):
        #DATA ASSOCIATION *IS* THE KALMAN MEASUREMENT UPDATE!
        ### STEP 1: COURSE LEVEL ASSOCIATION
        #1. Cluster: EMST-EGBIS
        #   1a. Compute EMST over collection of points
        #   1b. Throw the EMST as input graph structure to EGBIS to compute clusters
        #   1c. Edge weights of EMST (euclidean distances between points) are taken directly
            #   as dissimilarity measure

        clusters = Cluster.cluster(data)
        #2. assign to static and dynamic background recursively with ICP (iterative closest point?)
        #   2a. clusters in C which contain measurements matched with boundary points in static background 
            # are associated with static background, and used to update or initialize new boundary points at fine level for static background
        #   2b. then, these clusters are removed from C, and similar process occurs recursively for each dynamic track (all other readings)
        #   2c. Clusters that remain in C at end of process are not associated with any track and each cluster will initialize
            # a tentative dynamic track.

            #ICP good bc it's after prediction step-- points will be in their predicted areas.
        
        ### STEP 2: FINE LEVEL ASSOCIATION
        #Assign to specific boundary points?
        #JCBB

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


class OdomUpdater:
    def __init__(self, dalpha, dbeta, dpsi):
        self.dalpha = dalpha
        self.dbeta = dbeta
        self.dpsi = dpsi
        self.R = np.eye(2)

    def update_sensor_mean_pose(self, prev_sensor_mean_pose, control_input):
        '''
        prev_pose is a 3x1 vector [alpha, beta, psi]        
        '''

        xs_new = np.zeros((2,1))
        psi = prev_sensor_mean_pose[2]
        self.R = Helper.compute_rot_matrix(psi-self.dpsi)
        delta_psi = control_input["delta_psi"]
        delta_l = control_input["delta_l"]
        xs_new[0] = prev_sensor_mean_pose[0:2].T+self.R @ (delta_psi*np.array([[self.dbeta, -self.dalpha]]).T+np.array([[0, delta_l]]).T)
        xs_new[1] = psi-delta_psi

        
        return xs_new

    def calc_F(self, prev_sensor_mean_pose, control_input):
        Fs = np.zeros((3,3))
        Fc = np.zeros((3,3))

        Fs[0:2, 0:2] = np.eye(2)
        psi = prev_sensor_mean_pose[2]


        delta_psi = control_input["delta_psi"]
        delta_l = control_input["delta_l"]

        B = self.R @ (delta_psi*np.array([[self.dalpha, self.dbeta]]).T+np.array([[0, delta_l]]).T)

        Fs[0:2, 2] = B
        Fs[2, :] = [0,0,1]

        Fc[0:2,0:2] = delta_psi*self.R @ np.array([[0,1], [-1,0]])
        Fc[0:2, 2] = -B
        Fc[2, :] = [0,0,0]

        F = np.hstack((Fs, Fc))

        return F

    def calc_G(self):
        G = np.zeros((2,2))
        G[0:2, 0] = self.R @ np.array([[1,0]]).T
        G[0:2, 1] = self.R @ np.array([[self.dbeta, -self.dalpha]]).T
        G[1, :] = [0,-1]

        return G
    
    def calc_Q(self):
        #see page 6 of paper..
        return Q


    def update_covariance(self, prev_sensor_mean_pose, control_input, P):
        F = self.calc_F(prev_sensor_mean_pose, control_input)
        G = self.calc_G()
        Q = self.calc_Q()

        P_new = np.zeros((P.shape))

        ##CHECK INDEXES FOR THE BELOW. THEY'RE ARBITRARY RN.
        P_new[0:2, 0:2] = F @ P[0:2,0:2]@ F.T + G @ Q @ G.T
        P_new[0:2, 2] = F @ P[0:2, 0:1]

        P_new[2,0:2] = P[0:1, 0:2] @ F.T
        P_new[2,2] = P[0:1, 0:1]

        return P_new
    
    
    def update(self, x, control_input, prev_P):

        prev_sensor_mean_pose = x["xs"] #check index
        
        xs_new = self.update_sensor_mean_pose(prev_sensor_mean_pose, control_input)
        P_new = self.update_covariance(prev_sensor_mean_pose, control_input, prev_P)

        x["xs"] = xs_new

        return x, P_new

class Tracker:
    lidarscan_topic = "/scan"
    odom_topic = "/odom"

    def __init__(self):
        self.init_publishers_subscribers()

        self.mean_state = [] #large state matrix
        self.P = [] #large covariance matrix

        self.prev_Odom_callback_time = rospy.Time.Now()
        self.L = 0.1 #dist between front and rear axles
        self.control_input = {"L": L}

        dalpha = None
        dbeta = None
        dpsi = None
        self.odom_updater = OdomUpdater(dalpha, dbeta, dpsi)
        self.lidarUpdater = lidarUpdater()

        self.x = {"xs": None, "xt": None, "xb": None, "xp": None, "xc": None}
        self.P = None


    def init_publishers_subscribers(self):
        self.lidar_sub = rospy.Subscriber(self.lidarscan_topic, LaserScan, self.lidar_callback, queue_size = 1)
        self.odom_sub = rospy.subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size = 1)

    def lidar_callback(self, data):
        dt = data.header.stamp - self.prev_Odom_callback_time
        self.lidarUpdater.update(self.x, dt, data)

    def odom_callback(self, data):
        """ void fx """
        dt = data.header.stamp - self.prev_Odom_callback_time
        self.prev_Odom_callback_time = data.header.stamp

        R = Helper.quat_to_rot(data.pose.pose.quaternion)

        theta = np.atan2(R[1,0], R[0,0])
        self.control_input["dt"] = dt
        self.control_input["theta"] = theta
        self.control_input["v"] = data.twist.twist.linear.x
        self.control_input["delta_l"] = data.twist.twist.linear.x*dt
        self.control_input["delta_psi"] = data.twist.twist.linear.x*dt*np.tan(theta)/self.L

        self.x, self.P = self.odom_updater.update(self.x, self.control_input, self.P)
        



    







        

if __name__ == "__main__":
    rospy.init_node("tracker")
    detector = Tracker()
    rospy.spin()