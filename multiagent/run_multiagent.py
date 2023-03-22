import os
# standard imports
from copy import deepcopy

# third party
import matplotlib
matplotlib.use("TKagg")

# commonroad-io
from commonroad.scenario.state import CustomState

# reactive planner
from commonroad_rp.utility.visualization import make_gif, visualize_multiagent_at_timestep

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

    # Initialize predictor
    predictor = ph.load_prediction(scenario, config.prediction.mode, config)


    #########################
    #   Initialize Agents   #
    #########################

    # List of all agents in the simulation
    agent_list = []
    for id in agent_id_list:
        agent_list.append(Agent(id, planning_problem_set.find_planning_problem_by_id(id),
                          scenario, config, os.path.join(log_path, f"{id}"),
                          mod_path))

    # List of all not yet terminated agents in the simulation
    running_agent_list = deepcopy(agent_list)

    # List of all agents that changed in the previous timestep
    outdated_agent_id_list = list()

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

        # Calculate predictions
        if config.prediction.mode:
            if config.prediction.mode == "walenet":
                # Calculate predictions for all obstacles using WaleNet.
                # The state is only used for accessing the current timestep.
                predictions = ph.main_prediction(predictor, scenario, [obs.obstacle_id for obs in scenario.obstacles],
                                                 running_agent_list[0].x_0, scenario.dt,
                                                 [float(config.planning.planning_horizon)])
            elif config.prediction.mode == "lanebased":
                print("Lane-based predictions are not supported for multiagent simulations.")
                predictions = None
            else:
                predictions = None
        else:
            predictions = None

        outdated_agent_id_list = [a.id for a in running_agent_list]

        # Step simulation
        # TODO parallelize
        for agent in running_agent_list:

            # Handle agents that join later
            if agent.current_timestep > current_timestep:
                continue

            print(f"[Simulation] Stepping Agent {agent.id}")

            # Simulate.
            status, dummy_obstacle = agent.step_agent(predictions)

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

        # Terminate agents
        for agent in terminated_agent_list:
            running_agent_list.remove(agent)

        # Synchronize agents
        for agent in running_agent_list:
            agent.update_scenario(outdated_agent_id_list, dummy_obstacle_list)

        # Update own scenario for predictions and plotting
        for id in outdated_agent_id_list:
            # manage agents that are not yet active
            if scenario.obstacle_by_id(id) is not None:
                scenario.remove_obstacle(scenario.obstacle_by_id(id))

        # Plot current frame
        if (config.debug.show_plots or config.debug.save_plots) and len(running_agent_list) > 0:
            visualize_multiagent_at_timestep(scenario, [a.planning_problem for a in running_agent_list],
                                             dummy_obstacle_list, current_timestep, config, log_path,
                                             traj_set_list=[a.planner.all_traj for a in running_agent_list],
                                             ref_path_list=[a.planner.reference_path for a in running_agent_list],
                                             predictions=predictions,
                                             plot_window=config.debug.plot_window_dyn)

        scenario.add_objects(dummy_obstacle_list)

        current_timestep += 1
        running = len(running_agent_list) > 0

    # TODO Evaluation
    print("[Simulation] Simulation completed.")

    # make gif
    if config.debug.gif:
        make_gif(config, scenario, range(0, current_timestep-1), log_path, duration=0.1)
