import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from numba import njit
from scipy.sparse import block_diag

##CUSTOM IMPORTS
from helper import Helper
from odomUpdater import OdomUpdater
from lidarUpdater import lidarUpdater

class Tracker:
    lidarscan_topic = "/scan"
    odom_topic = "/odom"

    def __init__(self, id):

        self.mean_state = [] #large state matrix
        self.P = [] #large covariance matrix

        self.prev_Odom_callback_time = time.time()
        self.prev_Lidar_callback_time = time.time()
        self.L = 0.1 #dist between front and rear axles
        self.control_input = {"L": L}

        dalpha = None
        dbeta = None
        dpsi = None
        self.odom_updater = OdomUpdater(dalpha, dbeta, dpsi)
        self.lidarUpdater = lidarUpdater()

        self.x = {"xs": None, "xt": None, "xb": None, "xp": None, "xc": None}
        self.P = None
        self.id = id


    def update(self, obs, time):
        if obs["LiDAR"]:
            self.lidar_callback(obs["scans"][self.id], time)
        if obs["Odom"]:
            self.odom_callback(obs, time)
    def lidar_callback(self, data, time):
        dt = time - self.prev_Lidar_callback_time
        self.prev_Lidar_callback_time = time
        self.lidarUpdater.update(self.x, dt, data)

    def odom_callback(self, data, time):
        """ void fx """
        dt = time - self.prev_Odom_callback_time
        self.prev_Odom_callback_time = time

        R = Helper.compute_rot_matrix(data["poses_theta"][self.id])

        theta = np.atan2(R[1,0], R[0,0])
        self.control_input["dt"] = dt
        self.control_input["theta"] = theta
        self.control_input["v"] = data["linear_vels_x"][self.id]
        self.control_input["delta_l"] = data["linear_vels_x"][self.id]*dt
        self.control_input["delta_psi"] = data["linear_vels_x"][self.id]*dt*np.tan(theta)/self.L

        self.x, self.P = self.odom_updater.update(self.x, self.control_input, self.P)




if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=3)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta],[conf.sx2, conf.sy2, conf.stheta2],[conf.sx3, conf.sy3, conf.stheta3]]))
    env.render()
    planner = PurePursuitPlanner(conf, 0.17145+0.15875)
    planner2 = PurePursuitPlanner(conf, 0.17145+0.15875)
    planner3 = PurePursuitPlanner(conf, 0.17145+0.15875)


    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        speed2, steer2 = planner2.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], work['tlad'], work['vgain'])
        speed3, steer3 = planner3.plan(obs['poses_x'][2], obs['poses_y'][2], obs['poses_theta'][2], work['tlad'],work['vgain'])


        obs, step_reward, done, info = env.step(np.array([[steer, speed],[steer2, speed2],[steer3, speed3]]))
        laptime += step_reward
        env.render(mode='human_fast')
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)