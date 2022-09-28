__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "0.5"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"


import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np

from commonroad.scenario.trajectory import Trajectory, State
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.common.solution import Solution, PlanningProblemSolution, VehicleModel, \
    VehicleType, CostFunction

from commonroad_dc.feasibility.feasibility_checker import trajectory_feasibility, VehicleDynamics, \
    state_transition_feasibility, position_orientation_objective, position_orientation_feasibility_criteria, _angle_diff

from commonroad_rp.configuration import Configuration


def create_planning_problem_solution(config: Configuration, state_list: List[State], scenario: Scenario,
                                     planning_problem: PlanningProblem):
    """
    Creates CommonRoad Solution object
    """
    ego_vehicle_trajectory = Trajectory(initial_time_step=state_list[0].time_step, state_list=state_list)
    pps = PlanningProblemSolution(planning_problem_id=planning_problem.planning_problem_id,
                                  vehicle_type=VehicleType(config.vehicle.cr_vehicle_id),
                                  vehicle_model=VehicleModel.KS,
                                  cost_function=CostFunction.WX1,
                                  trajectory=ego_vehicle_trajectory)

    # create solution object
    solution = Solution(scenario.scenario_id, [pps])
    return solution


def reconstruct_states(config: Configuration, states: List[State], inputs: List[State]):
    """reconstructs states from a given list of inputs by forward simulation"""
    vehicle_dynamics = VehicleDynamics.from_model(VehicleModel.KS, VehicleType(config.vehicle.cr_vehicle_id))

    x_sim_list = list()
    x_sim_list.append(states[0])
    for idx, inp in enumerate(inputs):
        x0, x0_ts = vehicle_dynamics.state_to_array(states[idx])
        u0 = vehicle_dynamics.input_to_array(inp)[0]
        x1_sim = vehicle_dynamics.forward_simulation(x0, u0, config.planning.dt, throw=False)
        x_sim_list.append(vehicle_dynamics.array_to_state(x1_sim, x0_ts+1))
    return x_sim_list


def reconstruct_inputs(config: Configuration, pps: PlanningProblemSolution):
    """
    reconstructs inputs for each state transition using the feasibility checker
    """
    vehicle_dynamics = VehicleDynamics.from_model(pps.vehicle_model, pps.vehicle_type)

    feasible_state_list = []
    reconstructed_inputs = []
    for x0, x1 in zip(pps.trajectory.state_list[:-1], pps.trajectory.state_list[1:]):
        # reconstruct inputs using optimization
        feasible_state, reconstructed_input_state = state_transition_feasibility(x0, x1, vehicle_dynamics,
                                                                                 config.planning.dt,
                                                                                 position_orientation_objective,
                                                                                 position_orientation_feasibility_criteria,
                                                                                 1e-8, np.array([2e-2, 2e-2, 3e-2]),
                                                                                 4, 100, False)
        feasible_state_list.append(feasible_state)
        reconstructed_inputs.append(reconstructed_input_state)
    return feasible_state_list, reconstructed_inputs


def plot_states(config: Configuration, state_list: List[State], save_path: str, reconstructed_states=None, plot_bounds=False):
    """
    Plots states of trajectory from a given state_list
    state_list must contain the following states: steering_angle, velocity, orientation and yaw_rate
    """
    plt.figure(figsize=(7, 7.5))
    plt.subplot(5, 1, 1)
    plt.plot([state.position[0] for state in state_list],
             [state.position[1] for state in state_list], color="black", label="planned")
    if reconstructed_states:
        plt.plot([state.position[0] for state in reconstructed_states],
                 [state.position[1] for state in reconstructed_states], color="blue", label="reconstructed")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(5, 1, 2)
    plt.plot(list(range(len(state_list))),
             [state.steering_angle for state in state_list], color="black", label="planned")
    if reconstructed_states:
        plt.plot(list(range(len(reconstructed_states))),
                 [state.steering_angle for state in reconstructed_states], color="blue", label="reconstructed")
    if plot_bounds:
        plt.plot([0, len(state_list)], [config.vehicle.delta_min, config.vehicle.delta_min],
                 color="red", label="bounds")
        plt.plot([0, len(state_list)], [config.vehicle.delta_max, config.vehicle.delta_max],
                 color="red")
    plt.ylabel("delta")
    plt.subplot(5, 1, 3)
    plt.plot(list(range(len(state_list))),
             [state.velocity for state in state_list], color="black", label="planned")
    if reconstructed_states:
        plt.plot(list(range(len(reconstructed_states))),
                 [state.velocity for state in reconstructed_states], color="blue", label="reconstructed")
    plt.legend()
    plt.ylabel("v")
    plt.subplot(5, 1, 4)
    plt.plot(list(range(len(state_list))),
             [state.orientation for state in state_list], color="black", label="planned")
    if reconstructed_states:
        plt.plot(list(range(len(reconstructed_states))),
                 [state.orientation for state in reconstructed_states], color="blue", label="reconstructed")
    plt.ylabel("theta")
    plt.tight_layout()
    plt.subplot(5, 1, 5)
    plt.plot(list(range(len(state_list))),
             [state.yaw_rate for state in state_list], color="black", label="planned")
    plt.ylabel("theta_dot")
    plt.tight_layout()

    # Save Output
    plot_path = os.path.join(save_path, "evaluation_plot_states")
    plt.savefig(f"{plot_path}.svg", format='svg')

    # plot errors in position, velocity, orientation
    if reconstructed_states:
        plt.figure(figsize=(7, 7.5))
        plt.subplot(3, 1, 1)
        plt.plot(list(range(len(state_list))), [abs(state_list[i].position[0] - reconstructed_states[i].position[0])
                                                for i in range(len(state_list))], color="black")
        plt.ylabel("pos_x error")
        plt.subplot(3, 1, 2)
        plt.plot(list(range(len(state_list))), [abs(state_list[i].position[1] - reconstructed_states[i].position[1])
                                                for i in range(len(state_list))], color="black")
        plt.ylabel("pos_y error")
        plt.subplot(3, 1, 3)
        plt.plot(list(range(len(state_list))), [abs(_angle_diff(state_list[i].orientation,
                                                                reconstructed_states[i].orientation))
                                                for i in range(len(state_list))], color="black")
        plt.ylabel("theta error")
        plt.tight_layout()

        # Save Output
        plot_path = os.path.join(save_path, "evaluation_plot_states_reconstructed")
        plt.savefig(f"{plot_path}.svg", format='svg')


def plot_inputs(config: Configuration, input_list: List[State], save_path: str, reconstructed_inputs=None, plot_bounds=False):
    """
    Plots inputs of trajectory from a given input_list
    input_list must contain the following states: steering_angle_speed, acceleration
    optionally plots reconstructed_inputs
    """
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(list(range(len(input_list))),
             [state.steering_angle_speed for state in input_list], color="black", label="planned")
    if reconstructed_inputs:
        plt.plot(list(range(len(reconstructed_inputs))),
                 [state.steering_angle_speed for state in reconstructed_inputs], color="blue",
                 label="reconstructed")
    if plot_bounds:
        plt.plot([0, len(input_list)], [config.vehicle.v_delta_min, config.vehicle.v_delta_min],
                 color="red", label="bounds")
        plt.plot([0, len(input_list)], [config.vehicle.v_delta_max, config.vehicle.v_delta_max],
                 color="red")
    plt.legend()
    plt.ylabel("v_delta")
    plt.subplot(2, 1, 2)
    plt.plot(list(range(len(input_list))),
             [state.acceleration for state in input_list], color="black", label="planned")
    if reconstructed_inputs:
        plt.plot(list(range(len(reconstructed_inputs))),
                 [state.acceleration for state in reconstructed_inputs], color="blue", label="reconstructed")
    if plot_bounds:
        plt.plot([0, len(input_list)], [-config.vehicle.a_max, -config.vehicle.a_max],
                 color="red", label="bounds")
        plt.plot([0, len(input_list)], [config.vehicle.a_max, config.vehicle.a_max],
                 color="red")
    plt.ylabel("a_long")
    plt.tight_layout()

    # Save Output
    plot_path = os.path.join(save_path, "evaluation_plot_inputs")
    plt.savefig(f"{plot_path}.svg", format='svg')
