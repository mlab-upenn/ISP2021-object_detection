import numpy as np
import rospy


class VirtualScanner:
    lidar_topic = "/scan"
    vscan_topic = "/vscan"
    def __init__(self):
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, LidarScan, self.lidar_callback, queue_size = 1)
        self.vscan_pub = rospy.Publisher(self.vscan_topic, VirtualScan, queue_size = 1)

    def lidar_callback(self):
