import numpy as np
from test_measurement_model import Measurement
#Rao Blackwell Particle Filter
num_particles = 100

##########
# 8 hidden paramters
# Structure: Rows 0-2: x,y,theta
#            Row 3: v
#            Row 4-7: W,L, Cx, Cy
##########

particle_states = 8 #number of hidden parameters. 

######
#is it 8, or less? see text: Thus, to keep computational complexity low, we
#turn to RBPFs [Doucet et al., 2000]. We estimate X and v by particles and keep
#Gaussian estimates for Ω within each particle.
########

class Filter:
    def __init__(self):
        self.particles = np.zeros((num_particles, particle_states))
        self.initialize_particles() #I think that you draw from prior distribution (is it uniform?)

    def initialize_particles(self):
        #do operations to self.particles
        pass

    def calc_motion_belief(self):
        pass

    def calc_geometry_belief(self):
        #search for geometry that maximizes gaussian representation of measurement likelihood
        #only recompute costs for rays directly affected by a lcal change in omega
        
        #when revising belief of vehicle's width and length, 
        #keep closest corner in place. Thus, changing W, and L, changes Cx and Cy.
        #Adjust Cx and Cy to keep anchor point in place
        pass

    def calc_importance_weights(self):


        pass

    def update(self):
        pass