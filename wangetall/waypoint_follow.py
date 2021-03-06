import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from numba import njit
import coarse_association
import cluster

import cleanupstates
from tempfile import TemporaryFile
outfile = TemporaryFile()
from collections import defaultdict

dynamic_tracks_dict = defaultdict(lambda:[])
def createDictDynamicTrack(no_tracks, Q):
    for track in range(no_tracks):
        for i in range(len(Q)):
            dynamic_tracks_dict[track].append(i)

    return dynamic_tracks_dict

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle



class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

    def load_waypoints(self, conf):
        # load waypoints
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle

def convert_scan_polar_cartesian(scan, angle):
    return np.sin(angle)*scan, np.cos(angle)*scan

if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta],[conf.sx2, conf.sy2, conf.stheta2]]))
    env.render()
    planner = PurePursuitPlanner(conf, 0.17145+0.15875)
    planner2 = PurePursuitPlanner(conf, 0.17145+0.15875)

    laptime = 0.0
    start = time.time()
    cl = cluster.Cluster()
    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        speed2, steer2 = planner2.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], work['tlad'], work['vgain'])

        num_beams = 1080
        fov = 4.7
        lidar_data, agent_x, agent_y = obs['scans'][0],obs['poses_x'][0], obs['poses_y'][0]
        theta = np.linspace(-fov/2., fov/2., num=num_beams)

        seed=123
        rng = np.random.default_rng(seed=seed)
        current_scan = lidar_data
        noise = rng.normal(0., 0.09, size=num_beams)
        Q_s = current_scan
        current_scan = current_scan + noise
        # print("Qs", Q_s[0])
        y, x = convert_scan_polar_cartesian(current_scan, theta)
        y_s, x_s = convert_scan_polar_cartesian(Q_s, theta)
        #plt.scatter(x, y)
        #plt.scatter(x_s, y_s)
        #print(np.max(current_scan))
        # print(theta, current_scan)
        #
        #plt.show()
        Q_s_cart = np.stack((x_s, y_s), axis=-1)
        # plt.scatter(agent_x, agent_y)
        # plt.scatter(x_s, y_s)
        #plt.scatter(Q_s_cart[0][:,0],Q_s_cart[0][:,1])
        Q_s_cart_noise = np.stack((x, y), axis=-1)

        cleanup = cleanupstates.CleanUpStates(Q_s_cart, agent_x, agent_y, lidar_range=30.0)
        cleaned = cleanup.run()
        cleanup_noise = cleanupstates.CleanUpStates(Q_s_cart_noise, agent_x, agent_y, lidar_range=30.0)
        cleaned_noise = cleanup_noise.run()


        dynamic = cleaned[np.where(np.logical_and(cleaned[:,1]>=1.16, cleaned[:,1]<=1.91))]
        dynamic = dynamic[np.where(np.logical_and(dynamic[:,0]>=3, dynamic[:,0]<=5))]
        #print(dynamic)
        #plt.scatter(dynamic[:,0],dynamic[:,1])
        static_background = np.delete(cleaned,np.where(np.logical_and(np.logical_and(cleaned[:,1]>=1.16, cleaned[:,1]<=1.91),np.logical_and(cleaned[:,0]>=3, cleaned[:,0]<=5))),0)
        #print(cleaned_wihtout_dyn)
        #plt.scatter(cleaned_wihtout_dyn[:,0],cleaned_wihtout_dyn[:,1])
        # plt.scatter(cleaned_noise[:,0],cleaned_noise[:,1])
        #plt.scatter(static_background[:,0],static_background[:,1])
        #plt.show()
        #print(lidar_data, agent_x, agent_y, obs['poses_theta'])
        #print(np.sin( obs['poses_theta']) * lidar_data)

        #2.: C <- CLUSTERMEASUREMENTS(Z)
        # plt.scatter(cleaned_noise[:,0],cleaned_noise[:,1])
        # plt.show()
        # with open('cleaned_with_noise.npy', 'wb') as f:
        #     np.save(f, cleaned_noise)
        # with open('cleaned_with_noise.npy', 'rb') as f:
        #     a = np.load(f)
        plt.scatter(static_background[:,0],static_background[:,1])
        plt.scatter(dynamic[:,0],dynamic[:,1])
        plt.show()
        C = cl.cluster(cleaned_noise)
        # for key in C.keys():
        #     P = cleaned_noise[C[key]]
        #     plt.scatter(P[:,0], P[:,1])
        # plt.show()
        ca = coarse_association.Coarse_Association(C)
        dynamic_tracks_dict = createDictDynamicTrack(1, dynamic)
        #print(dynamic_tracks_dict)
        A, A_d, new_tracks = ca.run(cleaned_noise, static_background, dynamic, dynamic_tracks_dict)
        for key in A.keys():
            P = cleaned_noise[A[key]]
            clusters_s = plt.scatter(P[:,0], P[:,1], marker = 'x')
        for key in A_d.keys():
            P_d = cleaned_noise[A_d[key]]
            clusters_d = plt.scatter(P_d[:,0], P_d[:,1], marker = '+')
        for key in new_tracks.keys():
            P_new = cleaned_noise[new_tracks[key]]
            clusters_new = plt.scatter(P_new[:,0], P_new[:,1], marker = 'o')
        car = plt.scatter(agent_x, agent_y)
        plt.legend((car, clusters_s, clusters_d, clusters_new),("car", "assigned static background", "assigned dynamic tracks", "new tracks"))
        plt.show()
        obs, step_reward, done, info = env.step(np.array([[0, 0],[steer2, speed2]]))
        laptime += step_reward
        env.render(mode='human')
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
