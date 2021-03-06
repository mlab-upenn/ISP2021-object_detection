import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from scipy.sparse import block_diag


import sys
sys.path.append(".") # Adds higher directory to python modules path.

##CUSTOM IMPORTS
from perception.helper import Helper
from perception.odomUpdater import OdomUpdater
# from lidarUpdater import lidarUpdater
from perception.cluster import Cluster
from pure_pursuit_planner import PurePursuitPlanner
import matplotlib.pyplot as plt
cl = Cluster()

class Test:
    def __init__(self):
        pass
    def test_cluster(self, obs):
        num_beams = 1080
        fov = 4.7
        lidar_data, agent_x, agent_y = obs['scans'],obs['poses_x'], obs['poses_y']
        theta = np.linspace(-fov/2., fov/2., num=num_beams)
        seed=123
        rng = np.random.default_rng(seed=seed)

        noise = rng.normal(0., 0.1, size=num_beams)
        current_scan = lidar_data + noise
        y, x = Helper.convert_scan_polar_cartesian(current_scan, theta)

        Q_s_cart = np.stack((x, y), axis=-1)[0]

        C = cl.cluster(Q_s_cart)
        for key in C.keys():
            P = Q_s_cart[C[key]]
    


def main():
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('maps/Melbourne/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()
    planner = PurePursuitPlanner(conf, 0.17145+0.15875)

    laptime = 0.0
    start = time.time()
    test = Test()
    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])

        test.test_cluster(obs)
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)




if __name__ == '__main__':
    main()

