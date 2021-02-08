import numpy as np
import rospy
from test_measurement_model import MeasurementModel 
from particle_filter import Filter


SCORE_THRESHOLD = 0.0

class Detector:
    vscan_topic = "/vscan"
    tracking_topic = "/tracking"
    def __init__(self):
        self.init_publishers_subscribers()
        self.vscan_dict = {0: None, 1: None, 2: None} #need 3 entries to detect vehicle 
        self.measModel = MeasurementModel()


    def init_publishers_subscribers(self):
        self.vscan_sub = rospy.Subscriber(self.vscan_topic, VirtualScan, self.vscan_callback, queue_size = 1)
        

        ##################################
        #TrackedVehicles message structure:
        #X = (x,y,theta)
        #v
        #omega = W, L, Cx, Cy
        ##################################

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
            vehicles = self.fit_vehicles(areas_to_focus)

            #then, estimate vehicle velocity with particle filter update step on second frame, score using measurement model
            velocities, filters = self.estimate_velocity(vehicles, frame = 0)
            #filter out the vehicles that have too low score

            vehicles, velocities, filters = self.elim_low_scorers(vehicles, velocities, filters, frame = 1)


            #then, do another particle filter update. Score that against the third frame.

            velocities, filters = self.estimate_velocity(vehicles, frame = 1)
            #filter out the vehicles that have too low score

            vehicles, velocities, filters = self.elim_low_scorers(vehicles, velocities, filters, frame = 2)

            #publish the tracked vehicles.


    def scan_difference(self):
        return None

    def detect_areas_to_focus(self, deltas):
        return None

    def fit_vehicles(self, areas_to_focus):
        #first, score
        for i in areas_to_focus.keys():
            score = self.measModel.calc_probabilities(areas_to_focus[i])
            #filter out the vehicles that have too low score
        #then, fit vehicles
        return fitted_vehicles
        
    def estimate_velocity(self, fitted_vehicles, frame):
        filter_list = []
        vel_list = []
        for vehicle in fitted_vehicles:
            fltr = Filter() #maybe pass in area of focus?
            fltr.update(self.vscan_dict[frame]) #update on first frame.
            vel = fltr.estimate_velocity()

            fltr_list.append(fltr)
            vel_list.append(vel)

        return vel_list, filter_list

    
    def elim_low_scorers(self, vehicles, velocities, filters, frame): #how to use frame?
        idx_list = []
        for index, vehicle in enumerate(vehicles):
            score = self.measModel.calc_probabilities(velocities, self.vscan_dict[frame]) #do I need to pass in pose here too?
            if score >= SCORE_THRESHOLD:
                idx_list.append(index)
        
        return vehicle[idx], velocities[idx], filters[idx]












        

if __name__ == "__main__":
    rospy.init_node("detector")
    detector = Detector()
    rospy.spin()