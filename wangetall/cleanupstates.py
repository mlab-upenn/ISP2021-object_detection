import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from numba import njit
from perception.helper import Helper
import math
import logging
from rdp import rdp

class CleanUpStates():
    def __init__(self):
        pass

    def run(self, lidar_center_x, lidar_center_y, lidar, state, lidar_range=10.0):
        self.lidar_center_x = lidar_center_x
        self.lidar_center_y = lidar_center_y
        self.lidar_range = lidar_range
        self.state = state
        self.removeOld()
        self.removeOutOfRangeAndOutOfView(lidar, state)
        # self.rdp()
        # how to remove obsucred tracks?
        #cleaned_points = self.removeObscured(valid_points_in_radius)


    def rdp(self):
        for idx, track in self.state.static_background.items():
            #plt.scatter(self.state.static_background.xb[:,0],self.state.static_background.xb[:,1], c="r", s=20, label="static before")
            #print("before:",len(self.state.static_background.xb))
            track.xb = rdp(track.xb, epsilon=0.01)
            #print("after:",len(self.state.static_background.xb))
            #plt.scatter(self.lidar_center_x, self.lidar_center_y, c="r", label="ego vehicle center")
            #plt.scatter(self.state.static_background.xb[:,0],self.state.static_background.xb[:,1], c="g", s=5, label="static")
            #plt.legend()
            #plt.show()
        for idx, track in self.state.dynamic_tracks.items():
            if track.xp.shape[0] > 20:
                plt.figure()
                plt.scatter(track.xp[:,0],track.xp[:,1], c="r", s=20, label="static before")
                # print("before:",track.xp.shape[0])

                # np.random.choice(boundary_points.shape[0], 100, replace=False)
                # prev_pts = np.copy(track.xp)
                new_pts = rdp(track.xp, epsilon=0.01)
                if new_pts.shape[0] > 10:
                    track.xp = new_pts
                plt.scatter(track.xp[:,0],track.xp[:,1], c="b", s=20, label="static after")
                plt.legend()

                plt.show()
                breakpoint()



    def removeOld(self):
        to_rm = []
        for idx, track in self.state.dynamic_tracks.items():
            if track.last_seen > track.seen_threshold:
                logging.info("Clean up states removing Track {}. Last seen {}".format(track.id, track.last_seen))
                to_rm.append(track.id)
        
        for track_id in to_rm:
            self.state.cull_dynamic_track(track_id)

    def removeOutOfRangeAndOutOfView(self, lidar, state):
        #check only if centroid is outside? easier? or check if some of the points of the track are outside?
        # if self.state.static_background.xb.size != 0:
        for idx, track in self.state.static_background.items():
            # print("---------static backgroud----------")
            # print("static size before:",self.state.static_background.xb.size)
            mask = (track.xb[:,0] - self.lidar_center_x)**2 + (track.xb[:,1] - self.lidar_center_y)**2 < self.lidar_range**2
            track.xb = track.xb[mask,:]
            # print("static size after:",self.state.static_background.xb.size)
            angles = []
            last = 0
            for i in range(int(track.xb.size/2)):
                #math.atan2 return [pi, -pi] ---> converting to [2pi, -2pi]
                angle = math.degrees(math.atan2(track.xb[i,1] - self.lidar_center_y, track.xb[i,0] - self.lidar_center_x))
                if(angle - math.degrees(self.state.xs[2])) <= -180:
                    angle = abs((angle - math.degrees(self.state.xs[2]))%180)
                else:
                    angle = abs((angle - math.degrees(self.state.xs[2])))
                angles.append(angle)
            #breakpoint()
            angles = np.array(angles)
            track.xb = track.xb[np.where(angles <= math.degrees(4.7/2))]
            print("static size after fov cleaning:",track.xb.size)

        for idx, track in list(self.state.dynamic_tracks.items()):
            mask = (track.kf.x[0] - self.lidar_center_x)**2 + (track.kf.x[1] - self.lidar_center_y)**2 < self.lidar_range**2
            if(mask == False):
                self.state.cull_dynamic_track(idx)
                logging.info("Track {}, outside of lidar_range.... removing. ".format(idx))
                continue
            angle = math.degrees(math.atan2(track.kf.x[1] - self.lidar_center_y, track.kf.x[0]- self.lidar_center_x))
            if(angle - math.degrees(self.state.xs[2])) <= -180:
                angle = abs((angle - math.degrees(self.state.xs[2]))%180)
            else:
                angle = abs((angle - math.degrees(self.state.xs[2])))
            #breakpoint()
            if(angle > math.degrees(4.7/2)):
                self.state.cull_dynamic_track(idx)
                logging.info("Track {}, outside of field of view (in angle {}),.... removing".format(idx, angle))

            #self.state.static_background.xb = self.state.static_background.xb[np.where(abs(rads - self.state.xs[2]) <= 4.7/2)]
            #print("static size after fov cleaning:",self.state.static_background.xb.size)
        # for idx, track in list(self.state.dynamic_tracks.items()):
        #     dynamic_P = track.xp+track.kf.x[0:2]
        #     plt.scatter(dynamic_P[:,0],dynamic_P[:,1],s=8, label="dynamic track")
        # plt.scatter(self.lidar_center_x, self.lidar_center_y, c="r", label="ego vehicle center")
        # plt.scatter(self.state.static_background.xb[:,0],self.state.static_background.xb[:,1], c="g", s=5, label="static")
        # plt.legend()
        # plt.show()


    def removeObscured(self, within_radius):
        lidar_center_x = self.lidar_center_x
        lidar_center_y = self.lidar_center_y
        within_radius_cleaned = within_radius
        for point1_x, point1_y in within_radius_cleaned:
            slope = (lidar_center_y - point1_y) / (lidar_center_x - point1_x)
            for point2_x, point2_y in within_radius_cleaned:
                if(point1_x != point2_x and point1_y != point2_y):
                    pt2_on = (point2_y - point1_y) == slope * (point2_x - point1_x)
                    pt2_between = (min(point1_x, lidar_center_x) <= point2_x <= max(point1_x, lidar_center_x)) and (min(point1_y, lidar_center_y) <= point2_y <= max(point1_y, lidar_center_y))
                    on_and_between = pt2_on and pt2_between
                    if(on_and_between):
                        to_del = np.stack((point2_x, point2_y), axis=-1)
                        array1 = np.where(within_radius_cleaned[:, 0] == [point1_x])
                        array2 = np.where(within_radius_cleaned[:, 1] == [point1_y])
                        within_radius_cleaned = np.delete(within_radius_cleaned, np.intersect1d(array1, array2), axis=0)

        return within_radius_cleaned


# First, we do some house-keeping where out-of-date dynamic tracks and boundary points on the static background
# that have fallen out of the sensor’s field of view are dropped.
if __name__ == "__main__":



    start = time.time()
    cleanup = CleanUpStates()
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
