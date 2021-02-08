import numpy as np
import rospy
from test_measurement_model import MeasurementModel 
from particle_filter import Filter

class Tracker:
    vscan_topic = "/vscan"
    tracking_topic = "/tracking"
    def __init__(self):
        self.init_publishers_subscribers()
        self.vscan_dict = {0: None, 1: None, 2: None} #need 3 entries to detect vehicle 
        self.measModel = MeasurementModel()
        self.pFilter = Filter()


    def init_publishers_subscribers(self):
        self.vscan_sub = rospy.Subscriber(self.vscan_topic, VirtualScan, self.vscan_callback, queue_size = 1)
        
        ###TrackedVehicles message contains Centroids of vehicles. What else? 
        self.tracking_sub = rospy.Subscriber(self.tracking_topic, TrackedVehicles, self.tracking_callback, queue_size = 1)
    def vscan_callback(self, data):
        pass



    def tracking_callback(self, data):
        pass        



        

if __name__ == "__main__":
    rospy.init_node("tracker")
    tracker = Tracker()
    rospy.spin()