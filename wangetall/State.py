import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import sys
class State:
    def __init__(self):
        self.dynamic_tracks = {}
        self.static_background = StaticTrack(0, status=1)

        # self.laserpoints = [] #if in ROS, laserpoints will be published.
<<<<<<< HEAD
        self.xs = np.zeros((3)) #xs ego vehicle pose-- [x, y, theta]; global frame
=======
        self.xs = np.zeros((5)) #xs ego vehicle pose-- [x, y, theta]; global frame
>>>>>>> rbenefo
        self.Pxs = np.eye(3) #covariance of ego vehicle pose
        self.xc = np.zeros((3)) #
        self.Pxc = np.eye(3)
            
    def create_new_track(self, laserpoints, clusterIds):
        if len(self.dynamic_tracks) == 0:
            idx = 1
        else:
            highest_id = max(self.dynamic_tracks.keys())
            idx = highest_id + 1

        boundary_points = laserpoints[clusterIds] #want boundarypoints to be in cartesian
        track = DynamicTrack(idx, status =0)
        track.kf.x = np.array([np.mean(boundary_points[:,0])+self.xs[0],
                            np.mean(boundary_points[:,1])+self.xs[1], 
                            0,
                            0,0,0])
        
        track.xp = np.array([boundary_points[:,0]-track.kf.x[0]+self.xs[0],
                            boundary_points[:,1]-track.kf.x[1]+self.xs[1]]).T
        self.dynamic_tracks[idx] = track
        return idx


    def cull_dynamic_track(self, idx):
        self.dynamic_tracks.pop(idx)
        
    def merge_tracks(self, track_id, target_id, kind):
        if kind == "dynamic":
            track = self.dynamic_tracks[track_id]
            target = self.dynamic_tracks[target_id]

            target.kf.x = (target.kf.x+track.kf.x)/2
            target.xp = np.append(target.xp, track.xp, axis = 0) #do I need to adjust their centerpoints?
        elif kind=="static":
            track = self.dynamic_tracks[track_id]
            self.static_background.xb = np.append(self.static_background.xb, track.xp+track.kf.x[0:2], axis = 0)
            #tracks are by default assumed to be dynamic. If they're being merged to the static boundary,
            #they are removed from list of dynamic objects.
        self.cull_dynamic_track(track_id)
    
    def num_dynamic_tracks(self):
        return len(self.dynamic_tracks)

class Track:
    mature_threshold = 3
    def __init__(self,idx, status):
        self.num_viewings = 1
        self.status = status #Status: 0, 1 --> tentative, confirmed
        ##Private attributes:
        self.id = idx

    def update_num_viewings(self):
        self.num_viewings += 1
        if self.status == 0:
            if self.num_viewings >= self.mature_threshold:
                self.status = not self.status #mark as confirmed
        

class DynamicTrack(Track):
    def __init__(self, idx, status):
        super().__init__(idx, status)
        """kind: Static: 0; Dynamic: 1"""
        """
        xt: [X,Y, Phi, Xdot, Ydot, Phidot] #world coordinates
        xp: [[X,Y]] #List of boundary points in local coords
        """
        self.kind = 1
        self.xp = None
        self.kf = ExtendedKalmanFilter(dim_x=6, dim_z=6)


class StaticTrack(Track):
    def __init__(self, idx, status):
        super().__init__(idx, status)
        """kind: Static: 0; Dynamic: 1"""
        """
        xt: [X,Y, Phi, Xdot, Ydot, Phidot] #world coordinates
        xb: [[X,Y]] #List of boundary points in local coords
        """
        self.kind = 0
        self.xb = np.zeros((0,2))
        self.kf = ExtendedKalmanFilter(dim_x=2, dim_z=2)
