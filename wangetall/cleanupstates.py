import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from numba import njit

class CleanUpStates():
    def __init__(self):
        pass

    def run(self, lidar_center_x, lidar_center_y, lidar, state, lidar_range=30.0):
        self.lidar_center_x = lidar_center_x
        self.lidar_center_y = lidar_center_y
        self.lidar_range = lidar_range
        self.state = state

        self.removeOutOfRange(lidar, state)

        # how to remove obsucred tracks?
        #cleaned_points = self.removeObscured(valid_points_in_radius)



    def removeOutOfRange(self, lidar, state):
        #check only if centroid is outside? easier? or check if some of the points of the track are outside?
        if self.state.static_background.xb.size != 0:
            #print("---------static backgroud----------")
            #print(self.state.static_background.xb.size)
            mask = (self.state.static_background.xb[:,0] - self.lidar_center_x)**2 + (self.state.static_background.xb[:,1] - self.lidar_center_y)**2 < self.lidar_range**2
            self.state.static_background.xb = self.state.static_background.xb[mask,:]
            #print(self.state.static_background.xb.size)
        for idx, track in list(self.state.dynamic_tracks.items()):
            mask = (track.kf.x[0] - self.lidar_center_x)**2 + (track.kf.x[1] - self.lidar_center_y)**2 < self.lidar_range**2
            if(mask == False):
                self.state.cull_dynamic_track(idx)
                print("Track", idx, "outside of lidar_range.... removing")
            #within_radius = track.kf.x[mask,:]

            #plt.scatter(track.kf.x[0], track.kf.x[1], color="purple", label="Dynamic Centroid")
            #plt.scatter(within_radius[:,0], within_radius[:,1], c="g", label="within radius", s=8 )
        # for idx, track in self.state.dynamic_tracks.items():
        #     print(idx)
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
                        print("delete",to_del)
                        array1 = np.where(within_radius_cleaned[:, 0] == [point1_x])
                        array2 = np.where(within_radius_cleaned[:, 1] == [point1_y])
                        print(np.where(within_radius_cleaned[:, 0] == [point1_x]))
                        print(np.where(within_radius_cleaned[:, 1] == [point1_y]))
                        print(np.intersect1d(array1, array2))
                        within_radius_cleaned = np.delete(within_radius_cleaned, np.intersect1d(array1, array2), axis=0)

        return within_radius_cleaned


# First, we do some house-keeping where out-of-date dynamic tracks and boundary points on the static background
# that have fallen out of the sensor’s field of view are dropped.
if __name__ == "__main__":



    start = time.time()
    cleanup = CleanUpStates()
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
