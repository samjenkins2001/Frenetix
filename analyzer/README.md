# Interactive Debugging tool

## Requirements
Same requirements as for the commonroad-reactive-planner apply.

## Installation
* Adapt changes made in `commonroad_rp/utility/visualization.py` 
* Clone `/analyzer` file

## Run Code

* Run `analyzer/amazing_visualizer.py` to visualize logged data on a dashboard

## Tool Guide

### Main window

The main window of the analyzer displays the scenario and allows to retrace what happened in the planner at each time step.

<img src="Pictures/scenario.png" title= "Scenario" width="400" height="400">

### Cost distribution window for chosen trajectory

The second plot shows the cost distribution of the chosen trajectory number of the current time step.

<img src="Pictures/removed.png" title= "Cost distribution of chosen trajectories" width="400" height="200">

### Cost distribution window displaying all trajectories

The third plot displays the cost distribution for all trajectories of the current time step and therefore, allows to compare the trajectories.

<img src="Pictures/2.png" title= "Cost distributions" width="400" height="200">

### The following (interactive) features allow a more extensive analysis of the single parts.

Hover over images to access additional information.

1. Input fields to choose time step and trajectory number of the displayed scenario.
    * The first example displays the scenario for the chosen time step 31 and highlights the chosen trajectory number 61.
    * The different costs for following a trajectory are expressed by colors. 
        * Red indicates high costs, green low costs. The goal of this is that one gets a quick overview of the total costs of each trajectory. This can be changed easily to any other cost variables such as prediction costs or velocity costs. As the trajectories are ordered by their total costs, the trajectory number 0 will be colored in deepest green, while the highest trajectory number will be colored in the deepest red.
    * In the second example the costs of each trajectory were changed to the euclidean distance between the end point of the trajectory and the ego vehicle position. Hence, trajectories with an end point very close to the ego vehicle position are colored green and the ones far away from the ego vehicle position are colored red.

<img src="Pictures/input.png" title= "Input fields" width="400" height="350">
<img src="Pictures/costs.png" title= "Color scale" width="400" height="350">

2. Feasibility filter
    * One can choose between displaying all or only the feasible or infeasible trajectories, refer to the first two following visualizations.
    * If a trajectory is chosen that does not belong to the chosen feasibility, the next higher trajectory number that fullfills the feasibility criteria is chosen. An illustration is given below: For example if the lowest trajectory number that is feasible is the number 67. When choosing any lower trajectory number, the tool will automatically set the trajectory number to 67 as this is the next higher trajectory number fullfilling the chosen feasibility criteria as can be seen in the third visualization. 
    * Additionally, when choosing only feasible or infeasible trajectories, the plot only displays the cost distribution for the trajectories fullfilling the feasibility criteria which is displayed in the fourth visualization.

<img src="Pictures/all.png" title= "All trajectories" width="400" height="350">
<img src="Pictures/feasible.png" title= "Feasible trajectories" width="400" height="350">
<img src="Pictures/67.png" title= "Automatic correction" width="400" height="350">

3. Zooming, Pan and Hovering Controls
    * According to the Plotly documentation https://plotly.com/chart-studio-help/zoom-pan-hover-controls/


4. Disabling selected cost factors
    * By clicking the name of a cost factor in one of the two bar plots, the cost factor can be removed and added again to the plot and restrict the analysis to the remaining ones. This allows to have a closer look on some of the cost factors.

<img src="Pictures/0.png" title= "All cost factors" width="400" height="200">
<img src="Pictures/1.png" title= "jerk_lat_costs disabled" width="400" height="200">
<img src="Pictures/2.png" title= "jerk_lat_costs and prediction costs disabled" width="400" height="200">


### Dashboards 

Additional dashboards display information about the current time step such as positions, velocities and acceleration as well as infeasibility reasons and number of trajectories that were infeasible due to those reasons. 

## Literature
[1] Werling M., et al. *Optimal trajectory generation for dynamic street scenarios in a frenet frame*. In: IEEE International Conference on Robotics and Automation, Anchorage, Alaska, 987–993.

[2] Werling M., et al. *Optimal trajectories for time-critical street scenarios using discretized terminal manifolds* In:
The International Journal of Robotics Research, 2012
