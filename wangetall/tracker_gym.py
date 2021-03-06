import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from scipy.sparse import block_diag

##CUSTOM IMPORTS
from perception.helper import Helper
from perception.odomUpdater import OdomUpdater
from perception.lidarUpdater import lidarUpdater

from pure_pursuit_planner import PurePursuitPlanner
from State import State

class Tracker:
    def __init__(self, idx):

        self.prev_Odom_callback_time = time.time()
        self.prev_Lidar_callback_time = time.time()
        self.L = 0.1 #dist between front and rear axles
        self.control_input = {"L": L}

        dalpha = 0
        dbeta = 0
        dpsi = 0
        self.odom_updater = OdomUpdater(dalpha, dbeta, dpsi)
        self.lidarUpdater = lidarUpdater()

        self.id = idx
        self.state = State()
        self.state.xs = np.array([0,0])
        
    def update(self, obs, time):
        if obs["LiDAR"]:
            self.lidar_callback(obs["scans"][self.id], time)
        if obs["Odom"]:
            self.odom_callback(obs, time)
    
    def lidar_callback(self, data, time):
        dt = time - self.prev_Lidar_callback_time
        self.prev_Lidar_callback_time = time
        self.lidarUpdater.update(dt, data, self.state)

    def odom_callback(self, data, time):
        dt = time - self.prev_Odom_callback_time
        self.prev_Odom_callback_time = time

        R = Helper.compute_rot_matrix(data["poses_theta"][self.id])

        theta = np.atan2(R[1,0], R[0,0])
        self.control_input["dt"] = dt
        self.control_input["theta"] = theta
        self.control_input["v"] = data["linear_vels_x"][self.id]
        self.control_input["delta_l"] = data["linear_vels_x"][self.id]*dt
        self.control_input["delta_psi"] = data["linear_vels_x"][self.id]*dt*np.tan(theta)/self.L

        self.odom_updater.update(self.control_input, self.state)




if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=3)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta],[conf.sx2, conf.sy2, conf.stheta2]]))
    env.render()
    tracker = Tracker(0)
    planner = PurePursuitPlanner(conf, 0.17145+0.15875)


    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = 0,0 #ego vehicle with tracker
        speed2, steer2 = planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], work['tlad'], work['vgain']) #target vehicle


        obs, step_reward, done, info = env.step(np.array([[steer, speed],[steer2, speed2]]))
        laptime += step_reward
        env.render(mode='human_fast')
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)