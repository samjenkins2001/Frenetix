# Reactive Planner

Currently, this project generates solutions to trajectory planning problems given in the [CommonRoad](https://commonroad.in.tum.de/) scenario format.
The trajectories are generated according to the sampling-based approach in [1][2]. 

## Getting Started
These instructions should help you to install the trajectory planner and use it for development and testing purposes.

### Requirements
The software is  developed and tested on recent versions of Linux and OS X.

For the python installation, we suggest the usage of Virtual Environment with Python 3.8.
For the development IDE we suggest [PyCharm](http://www.jetbrains.com/pycharm/)

### Installation
1. Clone this repository, `checkout develop_RT`  & create a new virtual environment `python3.8 -m venv venv`


2. Install the package:
    * Source & Install the package via pip: `source venv/bin/activate` & `pip install -r requirements.txt`


### Example script

An example script `run_planner.py` is provided, which plans intended trajectories for motion planning. 


## Literature
[1] Werling M., et al. *Optimal trajectory generation for dynamic street scenarios in a frenet frame*. In: IEEE International Conference on Robotics and Automation, Anchorage, Alaska, 987–993.

[2] Werling M., et al. *Optimal trajectories for time-critical street scenarios using discretized terminal manifolds* In:
The International Journal of Robotics Research, 2012
