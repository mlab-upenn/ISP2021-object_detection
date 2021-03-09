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

    def run(self, Q_s, lidar_center_x, lidar_center_y, lidar_range):
        self.Q_s = Q_s
        self.lidar_center_x = lidar_center_x
        self.lidar_center_y = lidar_center_y
        self.lidar_range = lidar_range
        valid_points_in_radius = self.removeOutOfRange()

        #cleaned_points = self.removeObscured(valid_points_in_radius)

        return valid_points_in_radius


    def removeOutOfRange(self):
        mask = (self.Q_s[:,0] - self.lidar_center_x)**2 + (self.Q_s[:,1] - self.lidar_center_y)**2 < self.lidar_range**2
        within_radius = self.Q_s[mask,:]

        return within_radius

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
