import numpy as np
import rospy


lidar_scan_time = None #TODO: FILL WITH LIDAR SCAN TIME (HOW LONG IT TAKES TO COMPLETE A SINGLE SCAN)

class VirtualScanner:
    lidar_topic = "/scan"
    vscan_topic = "/vscan"
    odom_topic = "/odom"
    def __init__(self):
        self.init_publishers_subscribers()
        self.poses = np.zeros((4,100)) #Timestamp, X,Y, Heading?.
        self.pose_row_cnt = 0

        self.grid_dict = {"pose_origin": (0,0), "polar_grid": np.zeros((3, 1440))}
        

    def init_publishers_subscribers(self):
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, LidarScan, self.lidar_callback, queue_size = 1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size = 1)
        #how often does odom update? If LiDAR is updating faster than odom, we can't take average...
        self.vscan_pub = rospy.Publisher(self.vscan_topic, VirtualScan, queue_size = 1)

    def lidar_callback(self, data):

        self.generate_virtual_map(data)

        pass

    def odom_callback(self, data):
        self.pose[i] = [rospy.Time.Now(), data.pose] #TODO: PROCESS DATA
        self.pose_row_cnt += 1


    def generate_virtual_map(self, data):
        send_time = data.header.time
        begin_time = send_time - lidar_scan_time

        pose_idx = np.where(np.logical_and(self.poses[0]>= begin_time, self.poses[0]< send_time))
        locations = self.poses[pose_idx, 1:3]

        scan_origin = np.mean(locations, axis = 1)

        ranges = np.array(data.ranges)

        goodIdx = np.isfinite(ranges)
        angles = np.linspace(data.angle_min, data.angle_max, ranges.shape[0])
        
        ####################
        #Polar grid structure:
        #theta1 theta2 theta3....
        #isNAN? isNAN? isNAN? ....
        #closest closest closest ....
        ####################
        self.grid_dict["pose_origin"] = scan_origin
        self.grid_dict["polar_grid"][0] = angles #move to init?
        self.grid_dict["polar_grid"][1][goodIdx] = True
        self.grid_dict["polar_grid"][1][~goodIdx] = False
        self.grid_dict["polar_grid"][2][goodIdx] = ranges[goodIdx]

        self.pose_row_cnt = 0
        #clear self.poses

        pass

    def convert_grid_to_cartesian(self):
        """Called when we need to visualize the polar grid output. 
        Not actually used internally by the car
        
        May want to move this to another node, 
        as it'll likely be time consuming (good to run on separate thread)"""
        good_idx = np.where(self.grid_dict["pose_grid"][1])
        rel_x_coords = self.grid_dict["pose_grid"][good_idx]*np.cos(self.grid_dict["pose_grid"][0][good_idx])
        rel_y_coords = self.grid_dict["pose_grid"][good_idx]*np.sin(self.grid_dict["pose_grid"][0][good_idx])

        x_coords = rel_x_coords + self.grid_dict["pose_origin"][0]
        y_coords = rel_y_coords + self.grid_dict["pose_origin"][1]

       
        #How to visualize this in sim? Sim-dependent. May need to run Bresenham's line algorithm...
        return None

    def average_pose(self):
        pass



