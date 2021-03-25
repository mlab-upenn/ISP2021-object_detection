import numpy as np
from perception.helper import Helper
from scipy.linalg import block_diag
class OdomUpdater:
    def __init__(self, dalpha, dbeta, dpsi):
        self.dalpha = dalpha
        self.dbeta = dbeta
        self.dpsi = dpsi
        self.R = np.eye(2)

    def update_sensor_mean_pose(self, control_input):
        '''
        prev_pose is a 3x1 vector [alpha, beta, psi]        
        '''
        #how do we think about psi when sensor is rotating???
        prev_sensor_mean_pose = self.state.xs
        # xs_new = np.zeros((3))
        # psi = prev_sensor_mean_pose[2]
        # self.R = Helper.compute_rot_matrix(psi-self.dpsi)
        # delta_psi = control_input["delta_psi"]
        # delta_l = control_input["delta_l"]
        # xs_new[0:2] = prev_sensor_mean_pose[0:2].T+self.R @ (delta_psi*np.array([self.dbeta, -self.dalpha])+np.array([0, delta_l]))
        # xs_new[2] = psi-delta_psi


        #Since F1Tenth gym doesn't give info about wheel angle,
        # let's just update xs with actual xs to save time. 
<<<<<<< HEAD
        self.state.xs = np.array([control_input["poses_x"], control_input["poses_y"], control_input["poses_theta"]])
=======
        self.state.xs = np.array([control_input["poses_x"], control_input["poses_y"], \
            control_input["poses_theta"], control_input["linear_vels_x"]*np.cos(control_input["poses_theta"]), \
                control_input["linear_vels_x"]*np.sin(control_input["poses_theta"])])
>>>>>>> rbenefo


    def calc_F(self, prev_sensor_mean_pose, control_input):
        Fs = np.zeros((3,3))
        Fc = np.zeros((3,3))

        Fs[0:2, 0:2] = np.eye(2)
        psi = prev_sensor_mean_pose[2]


        delta_psi = control_input["delta_psi"]
        delta_l = control_input["delta_l"]

        B = self.R @ (delta_psi*np.array([self.dalpha, self.dbeta])+np.array([0, delta_l]))

        Fs[0:2, 2] = B
        Fs[2, :] = [0,0,1]

        Fc[0:2,0:2] = delta_psi*self.R @ np.array([[0,1], [-1,0]])
        Fc[0:2, 2] = -B
        Fc[2, :] = [0,0,0]

        # F = np.hstack((Fs, Fc))

        return Fs, Fc

    def calc_G(self):
        G = np.zeros((3,2))
        G[0:2, 0] = self.R @ np.array([1,0])
        G[0:2, 1] = self.R @ np.array([self.dbeta, -self.dalpha])
        G[2, :] = [0,-1]

        return G
    

    def calc_Q(self, control_input):
        #see page 6 of paper..
        theta = control_input["theta"]
        L = control_input["L"]
        U = np.array([[1, 0], [np.tan(theta)/L, control_input["v"]/(L*(np.cos(theta)**2))]])*control_input["dt"]
        V = np.eye(2)
        Q = U@V@U.T
        return Q


    def update_covariance(self, prev_sensor_mean_pose, control_input, state):
        Fs, Fc = self.calc_F(prev_sensor_mean_pose, control_input)
        # F = np.hstack((Fs, Fc))
        G = self.calc_G()
        Q = self.calc_Q(control_input)

        self.state.Pxs = Fs@self.state.Pxs @Fs.T + G@Q@G.T
        self.state.Pxc = Fc@self.state.Pxc @Fc.T + G@Q@G.T
    
    def update(self, control_input, state):
        self.state = state
        
        self.update_sensor_mean_pose(control_input)

        # self.update_covariance(prev_sensor_mean_pose, control_input, self.state)
