import os
# standard imports
import time
from copy import deepcopy

# third party
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib
matplotlib.use("TKagg")
from matplotlib import pyplot as plt

# commonroad-io
from commonroad_rp.utility.collision_report import coll_report

# commonroad-io
from commonroad.scenario.state import InputState, InitialState, CustomState

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner
# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, plot_final_trajectory, make_gif, visualize_scenario_and_pp
from commonroad_rp.utility.evaluation import create_planning_problem_solution, reconstruct_inputs, plot_states, \
    plot_inputs, reconstruct_states, create_full_solution_trajectory, check_acceleration
from commonroad_rp.cost_functions.cost_function import AdaptableCostFunction
from commonroad_rp.utility import helper_functions as hf

from commonroad_rp.utility.general import load_scenario_and_planning_problem
from commonroad_rp.configuration import Configuration

from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle, Circle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.planning.goal import GoalRegion
from commonroad.common.util import AngleInterval
from commonroad.common.util import Interval

from multiagent.agent import Agent

import commonroad_rp.prediction_helpers as ph
from behavior_planner.behavior_module import BehaviorModule


def run_multiagent(config: Configuration, log_path: str, mod_path: str):
    """ Adapted from commonroad_rp/run_planner.py
    Set up the configuration and the simulation.
    Creates and manages the agents.

    :param config: The configuration
    :param log_path: The path used for simulation-level logging,
                     agent-level logs are located in <log_path>/<agent_id>/
    :param mod_path: The path of the working directory of the planners
                     (containing planner configuration)
    """

    #########################################
    #   Load scenarios and configurations   #
    #########################################

    # Set configurations
    DT = config.planning.dt            # planning time step

    # Open the commonroad scenario and the planning problems of the
    # original ego vehicles.
    scenario, ego_planning_problem, original_planning_problem_set = load_scenario_and_planning_problem(config)

    # IDs of all dynamic obstacles that should be used as agents.
    # Add the dynamicObstacles to be simulated to the list.
    agent_id_list = config.multiagent.agent_ids
    # Planning problems for all agents.
    planning_problem_set = PlanningProblemSet()


    ###################################
    #   Initialize PlanningProblems   #
    ###################################

    # Create missing planning problems for additional agents
    # TODO parallelize
    for id in agent_id_list:
        obstacle = scenario.obstacle_by_id(id)
        initial_state = obstacle.initial_state
        if not hasattr(initial_state, 'acceleration'):
            initial_state.acceleration = 0.

        # create planning problem
        final_state = obstacle.prediction.trajectory.final_state
        goal_state = CustomState(time_step=Interval(final_state.time_step - 5, final_state.time_step + 5),
                                 position=Circle(1, final_state.position),
                                 velocity=Interval(final_state.velocity - 2, final_state.velocity + 2),
                                 orientation=AngleInterval(final_state.orientation - 0.349,
                                                           final_state.orientation + 0.349))

        problem = PlanningProblem(id, initial_state, GoalRegion(list([goal_state])))
        planning_problem_set.add_planning_problem(problem)


    # Add original ego vehicles to the simulation and the scenario.
    for problem in original_planning_problem_set.planning_problem_dict.values():
        id = problem.planning_problem_id
        agent_id_list.append(id)
        if not hasattr(problem.initial_state, 'acceleration'):
            problem.initial_state.acceleration = 0.
        planning_problem_set.add_planning_problem(problem)

        # add dummy obstacle for original ego vehicle
        # TODO add shape from vehicle params, here using values for BMW 320i
        dummy_obstacle = StaticObstacle(
            id,
            ObstacleType.CAR,
            Rectangle(length=4.508, width=1.61),
            problem.initial_state)

        scenario.add_objects(dummy_obstacle)


    #########################
    #   Initialize Agents   #
    #########################

    agent_list = []
    for id in agent_id_list:
        agent_list.append(Agent(id, planning_problem_set.find_planning_problem_by_id(id),
                          scenario, config, os.path.join(log_path, f"{id}"),
                          mod_path))

    running_agent_list = deepcopy(agent_list)
    running_agent_id_list = deepcopy(agent_id_list)

    # **************************
    # Run Planning
    # **************************

    # Step simulation as long as some agents have not completed

    current_timestep = 0

    running = True
    while running:
        # clear dummy obstacles
        dummy_obstacle_list = list()
        terminated_agent_list = list()

        # Step simulation
        # TODO parallelize
        for agent in running_agent_list:

            # Handle agents that join later
            if agent.current_timestep > current_timestep:
                continue

            print(f"[Simulation] Stepping Agent {agent.id}")

            # Simulate.
            status, dummy_obstacle = agent.step_agent()

            msg = ""
            if status > 0:
                if status == 1:
                    msg = "Completed."
                elif status == 2:
                    msg = "Failed to find valid trajectory."
                elif status == 3:
                    msg = "Collision detected."

                print(f"[Simulation] Agent {agent.id} terminated: {msg}")
                # Terminate all agents simultaneously
                terminated_agent_list.append(agent)
            else:
                # save dummy obstacle
                dummy_obstacle_list.append(deepcopy(dummy_obstacle))

        # Synchronize scenarios
        for id in running_agent_id_list:
            # manage agents that are not active
            if scenario.obstacle_by_id(id) is not None:
                scenario.remove_obstacle(scenario.obstacle_by_id(id))

        # Terminate agents
        for agent in terminated_agent_list:
            running_agent_list.remove(agent)
            running_agent_id_list.remove(agent.id)

        scenario.add_objects(dummy_obstacle_list)

        for agent in running_agent_list:
            agent.set_scenario(scenario)

        # Plot current frame
        rnd = MPRenderer(figsize=(20, 10))
        rnd.draw_params.time_begin = current_timestep
        scenario.draw(rnd)
        for problem in planning_problem_set.planning_problem_dict.values():
            problem.draw(rnd)
        rnd.render()
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir,
                        str(scenario.scenario_id) + "_{}.png".format(current_timestep)))

        current_timestep += 1
        running = len(running_agent_list) > 0

    # TODO Evaluation
    print("[Simulation] Simulation completed.")

    # make gif
    if config.debug.gif:
        make_gif(config, scenario, range(0, current_timestep), log_path, duration=0.1)
