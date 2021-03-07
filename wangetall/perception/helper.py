import numpy as np
class Helper:
    @staticmethod
    def compute_rot_matrix(angle):
        R = np.array([[np.cos(angle), -np.sin(angle)], 
                        [np.sin(angle), np.cos(angle)]])
        return R 

    @staticmethod
    def quat_to_rot(quat):
        R = np.zeros((3,3))
        R[0,0] = 2*(quat[0]**2+quat[1]**2)-1
        R[1,0] = 2*(quat[1]*quat[2]+quat[0]*quat[3])
        R[2,0] = 2*(quat[1]*quat[3]+quat[0]*quat[2])

        R[0,1] = 2*(quat[1]*quat[2]-quat[0]*quat[3])
        R[1,1] = 2*(quat[0]**2+quat[2]**2)-1
        R[2,1] = 2*(quat[2]*quat[3]+quat[0]*quat[1])
        R[0,2] = 2*quat(quat[1]*quat[3]+quat[0]*quat[2])
        R[1,2] = 2*(quat[2]*quat[3]-quat[0]*quat[1])
        R[2,2] = 2*(quat[0]**2+quat[3]**2)-1


        #can probs ignore the cells on the bottom right edge bc 2D
        return R
    @staticmethod
    def convert_scan_polar_cartesian(scan, angle):
        return np.sin(angle)*scan, np.cos(angle)*scan

    @staticmethod
    def convert_scan_polar_cartesian_joint(scan):
        return np.cos(scan[:,1])*scan[:,0], np.sin(scan[:,1])*scan[:,0]



# if __name__ =="__main__":
#     R = Helper.compute_rot_matrix(0.1)
#     x = np.array([1,1])
#     print(R)
#     print(R@x)
#     print(R@x.T)