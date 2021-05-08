# ISP2021-object_detection
This is the github project for the F1Tenth Independent Study Projects 2021. In this project we are focusing on the development of different approaches to achieve object detection and tracking based on lidar and camera.

## Requirements
- Linux Ubuntu (tested on versions 18.04.5 LTS)/ MacOS BigSur 11.2.3
- Python 3.7.4.
- F1Tenth Gym (tested on version 0.2)

## Installation
To install this repo please clone it to your local machine: 
````sh
git clone git@github.com:mlab-upenn/ISP2021-object_detection.git
cd ISP2021-object_detection/
````

Use the provided `requirements.txt` in the root directory of this repo, in order to install all required modules.\
`pip3 install -r requirements.txt`

The code is developed with Python 3.7.4.


## Running the code
````sh
cd wangetall/
python tracker_gym.py [-noplot]
````
Use ````-noplot```` to prevent the algorithm from displaying the plot information. Currently plotting is causing significant slowback of the visualisation of the algorithm. This option should be only used to check the real run-time of the algorithm. 




## Folder Structure

All main scripts depend on the following subfolders:

1. wangatall/ contains all the scripts and files needed to run the whole algorithm and is divided into following subfolders
    * perception/ contains core parts of the object detection algorithm


## wangetall folder content
| File | Description |
|----|----|
cleanupstates.py | part of algorithm for removal of unnecessary points for cleaning up memory
clearlogs.py | resets the log files
coarse_association.py | coarse association part of object tracking algorithm 
log.py | logger file
State.py | Maintains the state of the tracked objects
tracker_gym.py   | Is used to start the algorithm
tracker.py | Template for ROS version of the algorithm

## perception folder content
| File | Description |
|----|----|
cluster.py | clustering algorithm
helper.py | helper functions used troughout the code
icp.py | algorrithm used for coarse association
init_and_merge.py | algorithm to init and merge new tracks
jcbb_Cartesian.py | caresian version of jcbb
jcbb_numba.py | numba version of jcbb
jcbb.py | base version of jcbb
lidarUpdater.py | update lidar information
lidarUpdaterJCBB.py | update lidar information with numba
odomUpdater.py | update odometry information
walldetector.py | template for non-model free version of the algorithm for better recognition of walls

