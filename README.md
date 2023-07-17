# Reactive Planner

Currently, this project generates solutions to trajectory planning problems given in the [CommonRoad](https://commonroad.in.tum.de/) scenario format.
The trajectories are generated according to the sampling-based approach in [1][2].

## Getting Started
These instructions should help you to install the trajectory planner and use it for development and testing purposes.

### Requirements
The software is  developed and tested on recent versions of Linux and OS X.
For the python installation, we suggest the usage of Virtual Environment with Python 3.10.
For the development IDE we suggest [PyCharm](http://www.jetbrains.com/pycharm/)

### Installation
1. Make sure that the following dependencies are installed on your system:
   * [Eigen3](https://eigen.tuxfamily.org/dox/) 
     * On Ubuntu: `sudo apt-get install libeigen3-dev`
   * [Boost](https://www.boost.org/)
     * On Ubuntu: `sudo apt-get install libboost-all-dev`
   * [OpenMP](https://www.openmp.org/) 
     * On Ubuntu: `sudo apt-get install libomp-dev`

2. Clone this repository, `checkout develop_RT_cpp`  & create a new virtual environment `python3.10 -m venv venv`

3. Install the package:
    * Source & Install the package via pip: `source venv/bin/activate` & `pip install -r requirements.txt`

4. Download Scenarios:
    * Clone commonroad scenarios on the same level as commonroad-reactive-planner (not into commonroad-reactive-planner with: 
      * `git clone https://gitlab.lrz.de/tum-cps/commonroad-scenarios.git`

5. Download the latest wheel from the **FrenetPerformancePlannerFunctions**
    *  Go to the artifacts of the latest pipeline run: https://gitlab.lrz.de/motionplanning1/frenetperformanceplannerfunctions/-/pipelines and download the artifact
    *  unzip the artifact with `unzip artifacts.zip`
    *  install the wheel with `pip install frenetPlannerHelper*.whl`

6. Run the planner with `python3 main.py`

### Run Code
* An example script `run_planner.py` is provided, which plans intended trajectories for motion planning. Adjust path to select the scenario you want to execute.
* Change the configurations if you want to run a scenario with a different setup under `configurations/defaults/...` 
* If you want to execute a multiagent-simulation, please start `run_multiagent.py` 

## Literature
[1] Werling M., et al. *Optimal trajectory generation for dynamic street scenarios in a frenet frame*. In: IEEE International Conference on Robotics and Automation, Anchorage, Alaska, 987–993.

[2] Werling M., et al. *Optimal trajectories for time-critical street scenarios using discretized terminal manifolds* In:
The International Journal of Robotics Research, 2012
