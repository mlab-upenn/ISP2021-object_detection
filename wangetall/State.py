import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class State:
    def __init__(self):
        self.dynamic_tracks = {}
        self.static_background = Track(0, kind=1, status=1)

        self.laserpoints = [] #if in ROS, laserpoints will be published.
        self.xs = None
        self.xc = None
            
    def create_new_track(self, laserpoints, clusterIds):
        highest_id = max(self.dynamic_tracks.keys())
        idx = highest_id + 1

        boundary_points = laserpoints[clusterIds] #want boundarypoints to be in cartesian
        track = Track(idx, kind=None, status =0)

        track.kf.xt = np.array([np.mean(boundary_points[0]),
                            np.mean(boundary_points[1]), 
                            0,
                            0,0,0])
        track.xp = np.array([[boundary_points[0]-track.kf.xt[0]],
                            [boundary_points[1]-track.kf.xt[1]]])
        self.dynamic_tracks[idx] = track


    def cull_dynamic_track(self, id):
        self.dynamic_tracks.pop(id)
        
    def merge_tracks(self, track_id, target_id, kind):
        if kind == "dynamic":
            track = self.dynamic_tracks[track_id]
            target = self.dynamic_tracks[target_id]

            target.kf.xt = (target.kf.xt+track.kf.xt)/2
            target.xp = np.append(target.xp, track.xp) #do I need to adjust their centerpoints?
        elif kind=="static":
            track = self.dynamic_tracks[track_id]
            self.static_background.xp = np.append(self.static_background.xb, track.xp)
            
            #tracks are by default assumed to be dynamic. If they're being merged to the static boundary,
            #they are removed from list of dynamic objects.
        self.cull_dynamic_track(track_id)
    
    def num_dynamic_tracks(self):
        return len(self.dynamic_tracks)


class Track:
    def __init__(self, idx, kind, status):
        """kind: Static: 0; Dynamic: 1"""
        """
        xt: [X,Y, Phi, Xdot, Ydot, Phidot] #world coordinates
        xp: [[X,Y]] #List of boundary points in local coords
        """
        self.id = idx
        self.kind = kind
        self.xp = None
        self.xb = np.array([])
        self.kf = ExtendedKalmanFilter(dim_x=6, dim_z=2)
        self.status = status #Status: 0, 1 --> tentative, confirmed
        self.num_viewings = 0

        ##Private attributes:
        self.mature_threshold = 3
    
    def update_num_viewings(self):
        self.num_viewings += 1

        if self.status == 0:
            if self.num_viewings >= self.mature_threshold:
                self.status = not self.status #mark as confirmed
        
