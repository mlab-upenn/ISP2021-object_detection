import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from numba import jit
from perception.helper import Helper
import math
import logging
from rdp import rdp

class CleanUpStates():
    def __init__(self):
        pass

    def run(self, lidar_center_x, lidar_center_y, lidar, state, lidar_range=15.0):
        self.lidar_center_x = lidar_center_x
        self.lidar_center_y = lidar_center_y
        self.lidar_range = lidar_range
        self.state = state
        self.removeOld()
        self.removeOutOfRangeAndOutOfView()
        #self.rdp()
        # how to remove obsucred tracks?
        #cleaned_points = self.removeObscured(valid_points_in_radius)

    def rdp(self):
        if self.state.static_background.xb.size != 0:
            #plt.scatter(self.state.static_background.xb[:,0],self.state.static_background.xb[:,1], c="r", s=20, label="static before")
            #print("before:",len(self.state.static_background.xb))
            self.state.static_background.xb = rdp(self.state.static_background.xb, epsilon=0.01)
            #print("after:",len(self.state.static_background.xb))
            #plt.scatter(self.lidar_center_x, self.lidar_center_y, c="r", label="ego vehicle center")
            #plt.scatter(self.state.static_background.xb[:,0],self.state.static_background.xb[:,1], c="g", s=5, label="static")
            #plt.legend()
            #plt.show()


    def removeOld(self):
        to_rm = []
        for idx, track in self.state.dynamic_tracks.items():
            print("Clean up states removing track {}. Last seen {}".format(track.id, track.last_seen))
            if track.last_seen > track.seen_threshold:
                logging.info("Clean up states removing track {}. Last seen {}".format(track.id, track.last_seen))
                to_rm.append(track.id)

        for track_id in to_rm:
            #if track_id == 4:
                #breakpoint()
            self.state.cull_dynamic_track(track_id)

    @jit
    def removeOutOfRangeAndOutOfView(self):
        #check only if centroid is outside? easier? or check if some of the points of the track are outside?
        if self.state.static_background.xb.size != 0:
            mask = (self.state.static_background.xb[:,0] - self.lidar_center_x)**2 + (self.state.static_background.xb[:,1] - self.lidar_center_y)**2 < self.lidar_range**2
            self.state.static_background.xb = self.state.static_background.xb[mask,:]
            angles = []
            last = 0
            for i in range(int(self.state.static_background.xb.size/2)):
                #math.atan2 return [pi, -pi] ---> converting to [2pi, -2pi]
                angle = math.degrees(math.atan2(self.state.static_background.xb[i,1] - self.lidar_center_y, self.state.static_background.xb[i,0] - self.lidar_center_x))
                if(angle - math.degrees(self.state.xs[2])) <= -180:
                    angle = abs((angle - math.degrees(self.state.xs[2]))%180)
                else:
                    angle = abs((angle - math.degrees(self.state.xs[2])))
                angles.append(angle)
            #breakpoint()
            angles = np.array(angles)
            self.state.static_background.xb = self.state.static_background.xb[np.where(angles <= math.degrees(4.7/2))]
            logging.info("static size after fov cleaning:{}".format(self.state.static_background.xb.size))

        for idx, track in list(self.state.dynamic_tracks.items()):
            mask = (track.kf.x[0] - self.lidar_center_x)**2 + (track.kf.x[1] - self.lidar_center_y)**2 < self.lidar_range**2
            if(mask == False):
                self.state.cull_dynamic_track(idx)
                logging.info("Track, {}, outside of lidar_range.... removing. ".format(idx))
                continue
            angle = math.degrees(math.atan2(track.kf.x[1] - self.lidar_center_y, track.kf.x[0]- self.lidar_center_x))
            if(angle - math.degrees(self.state.xs[2])) <= -180:
                angle = abs((angle - math.degrees(self.state.xs[2]))%180)
            else:
                angle = abs((angle - math.degrees(self.state.xs[2])))
            #breakpoint()
            if(angle > math.degrees(4.7/2)):
                self.state.cull_dynamic_track(idx)
                logging.info("Track {} outside of field of view (in angle {}).... removing".format(idx, angle))

            #self.state.static_background.xb = self.state.static_background.xb[np.where(abs(rads - self.state.xs[2]) <= 4.7/2)]
            #print("static size after fov cleaning:",self.state.static_background.xb.size)
        # for idx, track in list(self.state.dynamic_tracks.items()):
        #     dynamic_P = track.xp+track.kf.x[0:2]
        #     plt.scatter(dynamic_P[:,0],dynamic_P[:,1],s=8, label="dynamic track")

        # plt.scatter(lidar[:,0], lidar[:,1], c="b", label="incoming lidar data")
        # plt.scatter(self.lidar_center_x, self.lidar_center_y, c="r", label="ego vehicle center")
        # #plt.scatter(self.state.static_background.xb[:,0],self.state.static_background.xb[:,1], c="g", s=5, label="static")
        # plt.legend()
        # plt.show()

# First, we do some house-keeping where out-of-date dynamic tracks and boundary points on the static background
# that have fallen out of the sensor’s field of view are dropped.
if __name__ == "__main__":



    start = time.time()
    cleanup = CleanUpStates()
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
