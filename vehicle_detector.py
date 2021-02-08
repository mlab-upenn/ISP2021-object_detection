import numpy as np
import rospy
from test_measurement_model import MeasurementModel 
from particle_filter import Filter

class Detector:
    vscan_topic = "/vscan"
    tracking_topic = "/tracking"
    def __init__(self):
        self.init_publishers_subscribers()
        self.vscan_dict = {0: None, 1: None, 2: None} #need 3 entries to detect vehicle 


    def init_publishers_subscribers(self):
        self.vscan_sub = rospy.Subscriber(self.vscan_topic, VirtualScan, self.vscan_callback, queue_size = 1)
        
        ###TrackedVehicles message contains Centroids of vehicles. What else? 
        self.tracking_pub = rospy.Publisher(self.tracking_topic, TrackedVehicles, queue_size = 1)
    def vscan_callback(self, data):
        emptykeys = sorted([k for k in self.vscan_dict.keys() if self.vscan_dict[k] is None])
        if len(emptykeys) > 0:
            self.vscan_dict[emptykeys[0]] = data
        else:
            ##shift the keys. Probably not very efficient...
            self.vscan_dict[0] = self.vscan_dict[1]
            self.vscan_dict[1] = self.vscan_dict[2]
            self.vscan_dict[2] = data

            #fit vehicle with importance sampling in area where change has been detected via scan differencing
            deltas = self.scan_difference()
            areas_to_focus = self.detect_areas_to_focus(deltas) #maybe return dict? 
            fitted_vehicles = self.fit_vehicles(areas_to_focus)

            #then, estimate vehicle velocity with particle filter update step on second frame, score using measurement model

            #then, do another particle filter update. Score that against the third frame.

            #filter out the vehicles that have too low score

            #publish the tracked vehicles.


    def scan_difference(self):
        return None

    def detect_areas_to_focus(self, deltas):
        return None

    def fit_vehicles(self, areas_to_focus):
        #first, score
        for i in areas_to_focus.keys():
            score = MeasurementModel.calc_probabilities(areas_to_focus[i])

        #then, fit vehicles
        return fitted_vehicles
        



        

if __name__ == "__main__":
    rospy.init_node("detector")
    detector = Detector()
    rospy.spin()