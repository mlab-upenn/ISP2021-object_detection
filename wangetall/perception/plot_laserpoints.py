import numpy as np
import matplotlib.pyplot as plt

def plot(data, graph_type):
    if graph_type =="polar":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.scatter(data[:,1], data[:,0], c="blue",marker = "o", alpha=0.5, label="Boundary Points")
        ax.set_xlim(np.pi, 1.5*np.pi)
        ax.set_ylim(0, 2)

        plt.show()
    else:
        plt.scatter(data[:,0], data[:,1], c = "blue", alpha= 0.5)
        plt.show()





if __name__ == "__main__":
    points = np.load("tests/npy_files/polar_laserscan.npy")
    plot(points, graph_type="polar")
