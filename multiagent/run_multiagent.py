import time
import warnings
from math import ceil
from multiprocessing import Queue
from queue import Empty

# commonroad-io
from commonroad.scenario.state import CustomState

# reactive planner
from commonroad_rp.utility.visualization import make_gif

from commonroad_rp.utility.general import load_scenario_and_planning_problem
from commonroad_rp.configuration import Configuration

from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle, Circle
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.planning.goal import GoalRegion
from commonroad.common.util import AngleInterval
from commonroad.common.util import Interval

from multiagent.agent import Agent
from multiagent.agent_batch import AgentBatch
from multiagent.multiagent_helpers import visualize_multiagent_at_timestep
from multiagent.multiagent_logging import *

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
    # Dummy obstacles for all agents.
    initial_obstacle_list = list()

    ###################################
    #   Initialize PlanningProblems   #
    ###################################

    for id in agent_id_list:
        obstacle = scenario.obstacle_by_id(id)
        initial_obstacle_list.append(obstacle)
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

    ###########################################################
    # Initialize PlanningProblem and Obstacle of original ego #
    ###########################################################

    for problem in original_planning_problem_set.planning_problem_dict.values():
        id = problem.planning_problem_id
        agent_id_list.append(id)
        if not hasattr(problem.initial_state, 'acceleration'):
            problem.initial_state.acceleration = 0.
        planning_problem_set.add_planning_problem(problem)

        # add dummy obstacle for original ego vehicle
        vehicle_params = config.vehicle
        dummy_obstacle = StaticObstacle(
            id,
            ObstacleType.CAR,
            Rectangle(length=vehicle_params.length, width=vehicle_params.width),
            problem.initial_state)

        scenario.add_objects(dummy_obstacle)
        initial_obstacle_list.append(dummy_obstacle)

    # Return values of the last agent step
    agent_state_dict = dict()
    for id in agent_id_list:
        agent_state_dict[id] = -1

    # Remove pending agents from the scenario
    for obs in initial_obstacle_list:
        if obs.initial_state.time_step > 0:
            scenario.remove_obstacle(obs)

    #############################
    #   Initialize Prediction   #
    #############################
    predictor = ph.load_prediction(scenario, config.prediction.mode, config)

    ################################
    #   Initialize Agent Batches   #
    ################################

    # List of tuples containing all batches and their associated fields:
    # [(AgentBatch,out_queue,in_queue,[ID])
    # batch_list[i][0]: Agent Batch object
    # batch_list[i][1]: Queue for sending data to the batch
    # batch_list[i][2]: Queue for receiving data from the batch
    # batch_list[i][3]: Agent IDs managed by the batch
    batch_list = []

    num_batches = 1
    if config.multiagent.multiprocessing:
        # We need at least one agent per batch, and one process for the main simulation
        num_batches = config.multiagent.num_procs-1
        if num_batches > len(agent_id_list):
            num_batches = len(agent_id_list)
    batch_size = ceil(len(agent_id_list) / num_batches)

    for i in range(num_batches):
        inqueue = Queue()
        outqueue = Queue()

        batch_list.append((AgentBatch(agent_id_list[i * batch_size:(i + 1) * batch_size],
                                      planning_problem_set, scenario,
                                      config, log_path, mod_path,
                                      outqueue, inqueue),
                           outqueue,
                           inqueue,
                           agent_id_list[i * batch_size:(i + 1) * batch_size]))

        batch_list[-1][0].start()

    ########################
    #      Run Planning    #
    ########################

    # Step simulation as long as some agents have not completed
    init_log(log_path)

    current_timestep = 0

    running = True
    while running:
        print(f"[Simulation] Simulating timestep {current_timestep}")
        # clear dummy obstacles
        dummy_obstacle_list = list()
        terminated_agent_list = list()

        # Calculate predictions
        if config.prediction.mode:
            if config.prediction.mode == "walenet":
                # Calculate predictions for all obstacles using WaleNet.
                # The state is only used for accessing the current timestep.
                predictions = ph.main_prediction(predictor, scenario, [obs.obstacle_id for obs in scenario.obstacles],
                                                 CustomState(time_step=current_timestep), scenario.dt,
                                                 [float(config.planning.planning_horizon)])
            elif config.prediction.mode == "lanebased":
                print("Lane-based predictions are not supported for multiagent simulations.")
                predictions = None
            else:
                predictions = None
        else:
            predictions = None

        # START TIMER
        step_time_start = time.time()

        ########################
        #    Step Simulation   #
        ########################

        # Send predictions
        for batch in batch_list:
            batch[1].put(predictions)

        # Receive results
        for batch in batch_list:
            try:
                dummy_obstacle_list.extend(batch[2].get(block=True, timeout=10))
                agent_state_dict.update(batch[2].get(block=True, timeout=10))
            except Empty:
                print("Timeout while waiting for step results! Exiting")
                return

        # Remove completed workers
        for batch in batch_list:
            if len(list(filter(lambda i: agent_state_dict[i] < 1, batch[3]))) == 0:
                print(f"[Simulation] Terminating batch {batch[3]}...")
                batch[1].put("END", block=True)
                batch[0].join()
                batch[1].close()
                batch[2].close()

                batch_list.remove(batch)

        # STOP TIMER
        step_time_end = time.time()

        # Update outdated agent lists
        outdated_agent_list = list(filter(lambda id: agent_state_dict[id] > -1, agent_id_list))

        # START TIMER
        sync_time_start = time.time()

        # Send dummy obstacles
        for batch in batch_list:
            batch[1].put(dummy_obstacle_list, block=True)
            batch[1].put(outdated_agent_list, block=True)

        # Update own scenario for predictions and plotting
        for id in filter(lambda i: agent_state_dict[i] >= 0, agent_state_dict.keys()):
            # manage agents that are not yet active
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not contained in the scenario")
                if scenario.obstacle_by_id(id) is not None:
                    scenario.remove_obstacle(scenario.obstacle_by_id(id))

        # STOP TIMER
        sync_time_end = time.time()

        # TODO Global plotting and logging
        """
        # Plot current frame
        if (config.debug.show_plots or config.debug.save_plots) and len(running_agent_list) > 0:
            visualize_multiagent_at_timestep(scenario, [a.planning_problem for a in running_agent_list],
                                             dummy_obstacle_list, current_timestep, config, log_path,
                                             traj_set_list=[a.planner.all_traj for a in running_agent_list],
                                             ref_path_list=[a.planner.reference_path for a in running_agent_list],
                                             predictions=predictions,
                                             plot_window=config.debug.plot_window_dyn)



        # remove terminated agents from outdated agents list
        for agent in terminated_agent_list:
            outdated_agent_id_list.remove(agent.id)

        append_log(log_path, current_timestep, current_timestep * scenario.dt,
                   step_time_end-step_time_start, sync_time_end-sync_time_start,
                   agent_id_list, [agent_state_dict[id] for id in agent_id_list])
        """

        scenario.add_objects(dummy_obstacle_list)

        current_timestep += 1
        running = len(batch_list) > 0

    print("[Simulation] Simulation completed.")
    print("Terminating workers...")

    # Workers should already have terminated, otherwise wait for timeouts
    for batch in batch_list:
        batch[0].join()

    # TODO make gif
    #if config.debug.gif:
    #    make_gif(config, scenario, range(0, current_timestep-1), log_path, duration=0.1)
