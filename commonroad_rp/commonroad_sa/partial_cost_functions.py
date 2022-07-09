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


def acceleration_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the acceleration cost for the given trajectory.
    """
    acceleration = trajectory.cartesian.a
    acceleration_sq = np.square(acceleration)
    cost = simps(acceleration_sq, dx=trajectory.dt)
    
    return cost


def jerk_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the jerk cost for the given trajectory.
    """
    acceleration = trajectory.cartesian.a
    jerk = np.diff(acceleration) / trajectory.dt
    jerk_sq = np.square(jerk)
    cost = simps(jerk_sq, dx=trajectory.dt)
    return cost


def jerk_lat_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the lateral jerk cost for the given trajectory.
    """
    jerk_sq_int = trajectory.trajectory_lat.squared_jerk_integral(trajectory.dt)
    cost = jerk_sq_int
    return cost


def jerk_lon_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the lateral jerk cost for the given trajectory.
    """
    jerk_sq_int = trajectory.trajectory_long.squared_jerk_integral(trajectory.dt)
    cost = jerk_sq_int
    return cost


def steering_angle_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the steering angle cost for the given trajectory.
    """
    raise NotImplementedError


def steering_rate_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the steering rate cost for the given trajectory.
    """
    raise NotImplementedError


def yaw_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the yaw cost for the given trajectory.
    """
    raise NotImplementedError


def lane_center_offset_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the Lane Center Offset cost.
    """
    raise NotImplementedError


def velocity_offset_cost(trajectory: commonroad_rp.trajectories.TrajectorySample, desired_speed, weights):
    """
    Calculates the Velocity Offset cost.
    """
    cost = np.sum((weights[0] * (trajectory.cartesian.v - desired_speed)) ** 2) + \
        (weights[1] * (trajectory.cartesian.v[-1] - desired_speed) ** 2) + \
        (weights[2] * (trajectory.cartesian.v[int(len(trajectory.cartesian.v) / 2)] - desired_speed) ** 2)
    return cost


def longitudinal_velocity_offset_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the Velocity Offset cost.
    """
    raise NotImplementedError


def orientation_offset_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the Orientation Offset cost.
    """
    cost = np.sum((np.abs(trajectory.curvilinear.theta)) ** 2) + (np.abs(trajectory.curvilinear.theta[-1])) ** 2
    return cost


def distance_to_reference_path_cost(trajectory: commonroad_rp.trajectories.TrajectorySample, desired_d, weights):
    """
    Calculates the Distance to Obstacle cost.
    """
    cost = np.sum((weights[0] * (desired_d - trajectory.curvilinear.d)) ** 2) + \
        (weights[1] * (desired_d - trajectory.curvilinear.d[-1])) ** 2
    return cost


def distance_to_obstacles_cost(trajectory: commonroad_rp.trajectories.TrajectorySample, timestep: int, scenario: Scenario):
    """
    Calculates the Distance to Obstacle cost.
    """
    cost = 0.0
    min_distance = 30.0
    pos_x = trajectory.cartesian.x[-1]
    pos_y = trajectory.cartesian.y[-1]
    for obstacle in scenario.dynamic_obstacles:
        state = obstacle.state_at_time(timestep)
        if state is not None:
            dist = np.sqrt((state.position[0] - pos_x)**2 + (state.position[1]-pos_y)**2)
            if dist < min_distance:
                cost += 1/dist
    return cost


def path_length_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the path length cost for the given trajectory.
    """
    velocity = trajectory.cartesian.v
    cost = simps(velocity, dx=trajectory.dt)
    return cost


def time_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the time cost for the given trajectory.
    """
    raise NotImplementedError


def inverse_duration_cost(trajectory: commonroad_rp.trajectories.TrajectorySample):
    """
    Calculates the inverse time cost for the given trajectory.
    """
    return 1 / time_cost(trajectory)

