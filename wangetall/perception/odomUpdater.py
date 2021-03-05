import numpy as np
from perception.helper import Helper

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
        #how do we think about psi when sensor is rotating???
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


    def update_covariance(self, prev_sensor_mean_pose, control_input, state):
        F = self.calc_F(prev_sensor_mean_pose, control_input)
        G = self.calc_G()
        Q = self.calc_Q()


        for _, track in self.state.dynamic_tracks.items():
            track.kf.G = G #there is no G. What to do about G?
            track.kf.F = F #there is no G. What to do about G?
            track.kf.Q = Q #there is no G. What to do about G?

            track.kf.predict()
        
        self.state.static_background.kf.F = F
        self.state.static_background.kf.Q = Q
        # self.state.static_background.kf.Q = G
        self.state.static_background.kf.predict()        

    
    
    def update(self, control_input, state):
        self.state = state
        prev_sensor_mean_pose = self.state.xs #check index
        
        xs_new = self.update_sensor_mean_pose(prev_sensor_mean_pose, control_input)
        self.state.xs = xs_new
        self.update_covariance(prev_sensor_mean_pose, control_input, self.state)
