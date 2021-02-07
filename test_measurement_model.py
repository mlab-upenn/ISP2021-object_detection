import numpy as np
import rospy

#X: vehicle pose
#omega: geometry
#D: virtual scan


#1.
#position rectangular shape representing vehicle according to X, Omega
#Build bounding box within predefined distance d_free around vehicle. 
    #If there's a vehicle, expect points within box to be occupied/occluded. Points in its vicinity, free/occluded.


class Measurement:
    lidar_topic = "/lidar"

    def __init__(self):
        self.p = None
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, LaserMsg, self.lidar_callback)

    def calc_eta(self):
        pass

    def calc_cost(self):
        pass
    
    def calc_var(self):
        pass

    def calc_probabilities(self):
        eta = self.calc_eta()
        ck  = self.calc_cost()
        var = self.calc_var()
        self.p = eta*np.exp(-ck**2/(2*var**2))

        return self.p
    
    def generate_map(self):
        #argmax of P?
        return map
    
    def lidar_callback(msg):
        pass
