# ISP2021-object_detection
This is the github project for the F1Tenth Independent Study Projects 2021. In this project we are focusing on the development of different approaches to achieve object detection and tracking based on lidar and camera.

## Requirements
- Linux Ubuntu (tested on versions 18.04.5 LTS)/ MacOS BigSur 11.2.3
- Python 3.7.4.
- F1Tenth Gym (tested on version 0.2)

## Installation
Use the provided `requirements.txt` in the root directory of this repo, in order to install all required modules.\
`pip3 install -r /path/to/requirements.txt`

The code is developed with Python 3.7.4.

## Running the code
````sh
cd wangetall

python tracker_gym.py [-noplot]
````
Use ````-noplot```` to prevent the algorithm from displaying the plot information. Currently plotting is causing significant slowback of the visualisation of the algorithm. This option should be only used to check the real run-time of the algorithm. 




## Folder Structure

All main scripts depend on the following subfolders:

1. wangatall/ contains all the scripts and files needed to run the whole algorithm and is divided into following subfolders
    * perception/ contains core parts of the object detection algorithm


## wanetall folder content
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
cluster.py | 
helper.py |
icp.py |
init_and_merge.py |
jcbb_Cartesian.py |
jcbb_numba.py |
jcbb.py |
lidarUpdater.py |
lidarUpdaterJCBB.py |
odomUpdater.py |
walldetector.py |

