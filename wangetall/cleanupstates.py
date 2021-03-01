class CleanUpStates():
    def __init__(self):
        self.Q_s = Q_s
        self.lidar_center = lidar_center
        self.lidar_range = lidar_range

    def run(self):
        valid_points_in_radius = removeOutOfRange()
        cleaned_points = removeObscured(valid_points_in_radius)


    def removeOutOfRange():
        mask = (self.Q_s[:,0] - self.lidar_center[:,0])**2 + (self.Q_s[:,1] - self.lidar_center[:,1])**2 < self.lidar_range**2
        within_radius = self.Q_s[mask,:]

        return within_raidus

    def removeObscured(within_raidus):
        lidar_center_x = self.lidar_center[:,0]
        lidar_center_y = self.lidar_center[:,1]
        for point1_x, point1_y in within_radius:
            slope = (lidar_center_y - point1_y) / (lidar_center_x - point1_x)
            for point2_x, point2_y in within_radius:
                if(point1_x != point2_x and point1_y != point2_y):
                    pt2_on = (point2_y - point1_y) == slope * (point2_x - point1_x)
                    pt2_between = (min(point1_x, lidar_center_x) <= point2_x <= max(point1_x, lidar_center_x)) and (min(point1_y, lidar_center_y) <= point2_y <= max(point1_y, lidar_center_y))
                    on_and_between = pt2_on and pt2_between
                    if(on_and_between):
                        print("point on and between")
                        print(point1_x, point1_y)
                        print(point2_x, point2_y)
                        print(lidar_center_x, lidar_center_y)
                        return point2_x, point2_y

# First, we do some house-keeping where out-of-date dynamic tracks and boundary points on the static background
# that have fallen out of the sensor’s field of view are dropped.
if __name__ == "__main__":
    cus = CleanUpStates()
