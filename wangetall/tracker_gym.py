import time
import sys
import yaml
import gym
import numpy as np
from argparse import Namespace
from scipy.sparse import block_diag
import matplotlib
import sys
import math
from timeit import default_timer as timer

matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt
from pylab import *
from matplotlib.patches import Arc

##CUSTOM IMPORTS
from perception.helper import Helper
from perception.odomUpdater import OdomUpdater
from perception.lidarUpdaterJCBB import lidarUpdater

from gym_testing.pure_pursuit_planner import PurePursuitPlanner
from State import State
import log

np.random.seed(10)
class Tracker:
    def __init__(self, idx, dt):

        self.prev_Odom_callback_time = time.time()
        self.prev_Lidar_callback_time = time.time()
        self.L = 0.1 #dist between front and rear axles
        self.control_input = {"L": self.L}

        dalpha = 0
        dbeta = 0
        dpsi = 0
        self.odom_updater = OdomUpdater(dalpha, dbeta, dpsi)
        self.lidarUpdater = lidarUpdater()

        self.id = idx
        self.state = State()
        self.dt = dt

    def update(self, obs, time):
        if obs["LiDAR"]:
            self.odom_callback(obs, time)
            self.lidar_callback(obs["scans"][self.id], time)

        if obs["Odom"]:
            self.odom_callback(obs, time)

    def lidar_callback(self, data, time):
        # dt = time - self.prev_Lidar_callback_time
        dt = self.dt*3 #LiDAR Update speed: 10ms --> 0.01s.
        self.prev_Lidar_callback_time = time
        self.lidarUpdater.update(dt, data, self.state)

    def odom_callback(self, data, time):
        # dt = time - self.prev_Odom_callback_time
        dt =self.dt
        self.prev_Odom_callback_time = time

        self.control_input["dt"] = dt
        # self.control_input["theta"] = theta
        # self.control_input["v"] = data["linear_vels_x"][self.id]
        # self.control_input["delta_l"] = data["linear_vels_x"][self.id]*dt
        # self.control_input["delta_psi"] = data["linear_vels_x"][self.id]*dt*np.tan(theta)/self.L


        self.control_input["poses_x"] = data["poses_x"][self.id]
        self.control_input["poses_y"] = data["poses_y"][self.id]
        self.control_input["poses_theta"] = data["poses_theta"][self.id]
        self.control_input["linear_vels_x"] = data["linear_vels_x"][self.id]
        self.control_input["linear_vels_y"] = data["linear_vels_y"][self.id]

        self.odom_updater.update(self.control_input, self.state)


def main(arg=None):
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('maps/Melbourne/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta], [conf.sx2, conf.sy2, conf.stheta2]]))

    show_env = False
    if show_env:
        env.render()
    planner = PurePursuitPlanner(conf, 0.17145+0.15875)
    planner2 = PurePursuitPlanner(conf, 0.17145+0.15875)
    planner3 = PurePursuitPlanner(conf, 0.17145+0.15875)

    laptime = 0.0
    start_total = time.time()

    tracker = Tracker(1,env.timestep)
    assert env.timestep == 0.01
    if arg == "-noplot":
        plot = False
    else:
        plot = True
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        #ax.set_xlim([-30, 30])
        #ax.set_ylim([-30,30])
    count = 0
    # breakpoint()
    tot_num_counts = 1000
    tracked_object_speeds = np.zeros((tot_num_counts,3))
    tracked_object_speeds[:,0] = np.arange(tot_num_counts)
    true_speeds = np.zeros((tot_num_counts))
    i = 0
    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        speed2, steer2 = planner2.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], work['tlad'], work['vgain'])
        tgt_speed_x = obs["linear_vels_x"][0]*np.cos(obs["poses_theta"][0])
        tgt_speed_y = obs["linear_vels_x"][0]*np.sin(obs["poses_theta"][0])
        tgt_speed = np.sqrt(tgt_speed_x**2+tgt_speed_y**2)
        true_speeds[i] = tgt_speed
        # print("Agent 2 speed {}".format(speed2))
        if count % 3 == 0 and count != 0:
            obs["Odom"] = False
            obs["LiDAR"] = True
        else:
            obs["Odom"] = True
            obs["LiDAR"] = False
        time_now = time.time()
        tracker.update(obs, time_now)
        if plot:
            fig.canvas.manager.window.move(0,0)
            ax.clear()
            ax.set_xlim([tracker.state.xs[0] - 10, tracker.state.xs[0] + 10])
            ax.set_ylim([tracker.state.xs[1] - 10, tracker.state.xs[1] + 10])
            static_background_state = tracker.state.static_background
            ax.scatter(static_background_state.xb[:,0], static_background_state.xb[:,1], color="black", label="Static Background", s=20)

            ax.scatter(tracker.state.xs[0], tracker.state.xs[1], color="blue")
            lidararc = Arc((tracker.state.xs[0], tracker.state.xs[1]), \
                width = 2, height = 2,\
                angle = math.degrees(tracker.state.xs[2]),\
                theta1= math.degrees(-4.7/2), theta2 = math.degrees(4.7/2), color="turquoise", linestyle="--",  alpha=0.8)
            ax.add_patch(lidararc)

            for idx, track in tracker.state.dynamic_tracks.items():
                ax.scatter(track.kf.x[0], track.kf.x[1], color="purple", s=60)
                ax.scatter(track.xp[:,0]+track.kf.x[0], track.xp[:,1]+track.kf.x[1], s = 1, c = track.color)
                trackspeed = round(np.sqrt(track.kf.x[3]**2+track.kf.x[4]**2), 2)                    
                if track.parent is not None:
                    ax.text(track.kf.x[0], track.kf.x[1], "T{} S:{}, P:{}".format(idx, trackspeed, track.parent), size = "x-small")
                else:
                    ax.text(track.kf.x[0], track.kf.x[1], "T{} S:{}".format(idx, trackspeed), size = "x-small")
                if track.id ==  13 or track.id == 24 or track.id == 131:
                    tracked_object_speeds[i,1:3] = [track.id, trackspeed]
            ax.scatter(obs["poses_x"][0], obs["poses_y"][0], c="orange",s = 5)
            plt.legend()

            plt.pause(0.00001)
        else:
            for idx, track in tracker.state.dynamic_tracks.items():
                trackspeed = round(np.sqrt(track.kf.x[3]**2+track.kf.x[4]**2), 2)
                if track.id ==  13 or track.id == 24 or track.id == 131:
                    tracked_object_speeds[i,1:3] = [track.id, trackspeed]
        
        np.save("Tracked_obj_speed.npy", tracked_object_speeds)
        np.save("true_speed.npy", true_speeds)
        i += 1
        print(i)
        if i > 999:
            break
        

        # plt.clf()


        obs, step_reward, done, info = env.step(np.array([[steer, speed], [steer2, speed2]]))
        laptime += step_reward
        if show_env:
            env.render(mode='human')
        count += 1
        end = timer()
    # print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start_total)


if __name__ == '__main__':
    log.get_logger()
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
    
