__author__ = "Alexander Hobmeier"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = []
__version__ = ""
__maintainer__ = "Alexander Hobmeier"
__email__ = "commonroad@lists.lrz.de"
__status__ = ""

import numpy as np
import commonroad_rp.trajectories
from commonroad.scenario.trajectory import State
from commonroad.scenario.scenario import Scenario
from scipy.integrate import simps
import commonroad_dc.pycrcc as pycrcc
from shapely.geometry import LineString, Point
from commonroad_rp.utility.helper_functions import distance


def acceleration_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the acceleration cost for the given trajectory.
    """
    acceleration = trajectory.cartesian.a
    acceleration_sq = np.square(acceleration)
    cost = simps(acceleration_sq, dx=trajectory.dt)
    
    return cost


def jerk_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the jerk cost for the given trajectory.
    """
    acceleration = trajectory.cartesian.a
    jerk = np.diff(acceleration) / trajectory.dt
    jerk_sq = np.square(jerk)
    cost = simps(jerk_sq, dx=trajectory.dt)
    return cost


def jerk_lat_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the lateral jerk cost for the given trajectory.
    """
    cost = trajectory.trajectory_lat.squared_jerk_integral(trajectory.dt)
    return cost


def jerk_lon_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the lateral jerk cost for the given trajectory.
    """
    cost = trajectory.trajectory_long.squared_jerk_integral(trajectory.dt)
    return cost


def steering_angle_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the steering angle cost for the given trajectory.
    """
    raise NotImplementedError


def steering_rate_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the steering rate cost for the given trajectory.
    """
    raise NotImplementedError


def yaw_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the yaw cost for the given trajectory.
    """
    raise NotImplementedError


def lane_center_offset_cost(trajectory: commonroad_rp.trajectories.TrajectorySample,
                    planner=None, scenario=None, desired_speed: float=0, weights=None) -> float:
    """
    Calculate the average distance of the trajectory to the center line of a lane.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        lanelet_network (LaneletNetwork): Considered lanelet network.

    Returns:
        float: Average distance from the trajectory to the center line of a lane.
    """
    dist = 0.0
    for i in range(len(trajectory.cartesian.x)):
        # find the lanelet of every position
        pos = [trajectory.cartesian.x[i], trajectory.cartesian.y[i]]
        lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([np.array(pos)])
        if len(lanelet_ids[0]) > 0:
            lanelet_id = lanelet_ids[0][0]
            lanelet_obj = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            # find the distance of the current position to the center line of the lanelet
            dist = dist + dist_to_nearest_point(lanelet_obj.center_vertices, pos)
        # theirs should always be a lanelet for the current position
        # otherwise the trajectory should not be valid and no costs are calculated
        else:
            dist = dist + 5

    return (dist / len(trajectory.cartesian.x)) * weights


def velocity_offset_cost(trajectory: commonroad_rp.trajectories.TrajectorySample,
                         planner=None, scenario=None,
                         desired_speed: float=0, weights=None) -> float:
    """
    Calculates the Velocity Offset cost.
    """
    cost = np.sum((weights[0] * (trajectory.cartesian.v - desired_speed)) ** 2) + \
        (weights[1] * (trajectory.cartesian.v[-1] - desired_speed) ** 2) + \
        (weights[2] * (trajectory.cartesian.v[int(len(trajectory.cartesian.v) / 2)] - desired_speed) ** 2)
    return cost


def longitudinal_velocity_offset_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the Velocity Offset cost.
    """
    raise NotImplementedError


def orientation_offset_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the Orientation Offset cost.
    """
    cost = np.sum((np.abs(trajectory.curvilinear.theta)) ** 2) + (np.abs(trajectory.curvilinear.theta[-1])) ** 2
    return cost


def distance_to_reference_path_cost(trajectory: commonroad_rp.trajectories.TrajectorySample,
                         planner=None, scenario=None,
                         desired_speed: float=0, weights=None) -> float:
    """
    Calculates the Distance to Reference Path costs.

    Args:
        trajectory (FrenetTrajectory): Frenét trajectory to be checked.

    Returns:
        float: Average distance of the trajectory to the given path.
    """
    # Costs of gerneral deviation from ref path
    cost = np.mean(np.abs(trajectory.curvilinear.d)) * weights[0]

    # Additional costs for deviation at final planning point from ref path
    cost += np.mean(np.abs(trajectory.curvilinear.d[-1])) * weights[1]

    return cost


def distance_to_obstacles_cost(trajectory: commonroad_rp.trajectories.TrajectorySample,
                         planner=None, scenario=None,
                         desired_speed: float=0, weights=None) -> float:
    """
    Calculates the Distance to Obstacle cost.
    """
    cost = 0.0
    min_distance = 30.0
    pos_x = [trajectory.cartesian.x[1], trajectory.cartesian.x[-1]]
    pos_y = [trajectory.cartesian.y[1], trajectory.cartesian.y[-1]]
    for obstacle in scenario.dynamic_obstacles:
        state = obstacle.state_at_time(planner.x_0.time_step)
        if state is not None:
            dist = np.sqrt((state.position[0] - pos_x[0])**2 + (state.position[1]-pos_y[0])**2)
            if dist < min_distance:
                cost += (dist - min_distance) ** 2
            dist = np.sqrt((state.position[0] - pos_x[1]) ** 2 + (state.position[1] - pos_y[1]) ** 2)
            if dist < min_distance:
                cost += (dist - min_distance) ** 2
    return cost * weights


def path_length_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the path length cost for the given trajectory.
    """
    velocity = trajectory.cartesian.v
    cost = simps(velocity, dx=trajectory.dt)
    return cost


def time_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the time cost for the given trajectory.
    """
    raise NotImplementedError


def inverse_duration_cost(trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
    """
    Calculates the inverse time cost for the given trajectory.
    """
    return 1 / time_cost(trajectory)


def velocity_costs(trajectory: commonroad_rp.trajectories.TrajectorySample,
                    planner=None, scenario=None,
                    desired_speed: float=0, weights=None) -> float:
    """
    Calculate the costs for the velocity of the given trajectory.

    Args:
        trajectory (FrenetTrajectory): Considered trajectory.
        ego_state (State): Current state of the ego vehicle.
        planning_problem (PlanningProblem): Considered planning problem.
        scenario (Scenario): Considered scenario.
        dt (float): Time step size of the scenario.

    Returns:
        float: Costs for the velocity of the trajectory.

    """
    # if the goal area is reached then just consider the goal velocity
    if reached_target_position(np.array([trajectory.cartesian.x[0], trajectory.cartesian.y[0]]), planner.goal_area):
        # if the planning problem has a target velocity
        if hasattr(planner.planning_problem.goal.state_list[0], "velocity"):
            return abs(
                (
                    planner.planning_problem.goal.state_list[0].velocity.start
                    + (
                        planner.planning_problem.goal.state_list[0].velocity.end
                        - planner.planning_problem.goal.state_list[0].velocity.start
                    )
                    / 2
                )
                - np.mean(trajectory.cartesian.v)
            )
        # otherwise prefer slow trajectories
        else:
            return np.mean(trajectory.cartesian.v)

    # if the goal is not reached yet, try to reach it
    # get the center points of the possible goal positions
    goal_centers = []
    # get the goal lanelet ids if they are given directly in the planning problem
    if (
        hasattr(planner.planning_problem.goal, "lanelets_of_goal_position")
        and planner.planning_problem.goal.lanelets_of_goal_position is not None
    ):
        goal_lanelet_ids = planner.planning_problem.goal.lanelets_of_goal_position[0]
        for lanelet_id in goal_lanelet_ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            n_center_vertices = len(lanelet.center_vertices)
            goal_centers.append(lanelet.center_vertices[int(n_center_vertices / 2.0)])
    elif hasattr(planner.planning_problem.goal.state_list[0], "position"):
        # get lanelet id of the ending lanelet (of goal state), this depends on type of goal state
        if hasattr(planner.planning_problem.goal.state_list[0].position, "center"):
            goal_centers.append(planner.planning_problem.goal.state_list[0].position.center)
    # if it is a survival scenario with no goal areas, no velocity can be proposed
    else:
        return 0.0

    # get the distances to the previous found goal positions
    distances = []
    for goal_center in goal_centers:
        distances.append(distance(goal_center, planner.x_0.position))

    # calculate the average distance to the goal positions
    avg_dist = np.mean(distances)

    # get the remaining time
    _, max_remaining_time_steps = calc_remaining_time_steps(
        planning_problem=planner.planning_problem,
        ego_state_time=planner.x_0.time_step,
        t=0.0,
        dt=trajectory.dt,
    )
    remaining_time = max_remaining_time_steps * trajectory.dt

    # if there is time remaining, calculate the difference between the average desired velocity and the velocity of the trajectory
    if remaining_time > 0.0:
        avg_desired_velocity = avg_dist / remaining_time
        avg_v = np.mean(trajectory.cartesian.v)
        cost = abs(avg_desired_velocity - avg_v)
    # if the time limit is already exceeded, prefer fast velocities
    else:
        cost = 30.0 - np.mean(trajectory.cartesian.v)

    return cost*weights


def reached_target_position(pos: np.array, goal_area):
    """
    Check if the given position is in the goal area of the planning problem.

    Args:
        pos (np.array): Position to be checked.
        goal_area (ShapeGroup): Shape group representing the goal area.

    Returns:
        bool: True if the given position is in the goal area.
    """
    # if there is no goal area (survival scenario) return True
    if goal_area is None:
        return True

    point = pycrcc.Point(x=pos[0], y=pos[1])

    # check if the point of the position collides with the goal area
    if point.collide(goal_area) is True:
        return True

    return False


def calc_remaining_time_steps(
    ego_state_time: float, t: float, planning_problem, dt: float
):
    """
    Get the minimum and maximum amount of remaining time steps.

    Args:
        ego_state_time (float): Current time of the state of the ego vehicle.
        t (float): Checked time.
        planning_problem (PlanningProblem): Considered planning problem.
        dt (float): Time step size of the scenario.

    Returns:
        int: Minimum remaining time steps.
        int: Maximum remaining time steps.
    """
    considered_time_step = int(ego_state_time + t / dt)
    if hasattr(planning_problem.goal.state_list[0], "time_step"):
        min_remaining_time = (
            planning_problem.goal.state_list[0].time_step.start - considered_time_step
        )
        max_remaining_time = (
            planning_problem.goal.state_list[0].time_step.end - considered_time_step
        )
        return min_remaining_time, max_remaining_time
    else:
        return False


def dist_to_nearest_point(center_vertices: np.ndarray, pos: np.array):
    """
    Find the closest distance of a given position to a polyline.

    Args:
        center_vertices (np.ndarray): Considered polyline.
        pos (np.array): Conisdered position.

    Returns:
        float: Closest distance between the polyline and the position.
    """
    # create a point and a line, project the point on the line and find the nearest point
    # shapely used
    point = Point(pos)
    linestring = LineString(center_vertices)
    project = linestring.project(point)
    nearest_point = linestring.interpolate(project)

    return distance(pos, np.array([nearest_point.x, nearest_point.y]))