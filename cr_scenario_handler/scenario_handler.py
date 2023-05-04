import time
import warnings
from math import ceil, isnan
from multiprocessing import Queue
from queue import Empty
from typing import Tuple
import random

# commonroad-io
from commonroad.scenario.state import CustomState
from commonroad.scenario.scenario import Scenario
from commonroad_prediction.prediction_module import PredictionModule

# reactive planner
from commonroad_rp.utility.visualization import make_gif

from cr_scenario_handler.utils.general import load_scenario_and_planning_problem
from commonroad_rp.configuration import Configuration

from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle, Circle
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.planning.goal import GoalRegion
from commonroad.common.util import AngleInterval
from commonroad.common.util import Interval

from cr_scenario_handler.agents.agent_batch import AgentBatch
import cr_scenario_handler.utils.multiagent_helpers as mh
from cr_scenario_handler.utils.multiagent_logging import *

import commonroad_rp.prediction_helpers as ph


def ScenarioHandler(config: Configuration, log_path: str, mod_path: str):
    """ Based on commonroad_rp/run_planner.py
    Set up the configuration and the simulation.
    Creates and manages the agents batches and their communication.

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
    all_obs_ids = mh.get_all_obstacle_ids(scenario)

    #########################################
    #    Select Obstacle IDs to Simulate    #
    #########################################
    if config.multiagent.use_specific_agents:
        agent_id_list = config.multiagent.agent_ids
        for agent_ids_check in agent_id_list:
            if agent_ids_check not in all_obs_ids:
                raise ValueError("Selected Obstacle IDs not existent in Scenario!\n"
                                 "Check selected 'agent_ids' in config!")
    else:
        agent_id_list = all_obs_ids

    if config.multiagent.number_of_agents < len(agent_id_list):
        if config.multiagent.select_agents_randomly:
            agent_id_list = random.sample(agent_id_list, config.multiagent.number_of_agents)
        else:
            agent_id_list = agent_id_list[:config.multiagent.number_of_agents]

    # Planning problems for all agents.
    planning_problem_set = PlanningProblemSet()
    # Dummy obstacles for all agents.
    initial_obstacle_list = []

    ########################################################
    #   Initialize PlanningProblems of additional Agents   #
    ########################################################

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

    if not config.multiagent.multiprocessing:
        agent_batch = AgentBatch(agent_id_list, planning_problem_set, scenario,
                                 config, log_path, mod_path,
                                 None, None)

        agent_batch.run_sequential(log_path, predictor, scenario)

    else:

        batch_list: List[Tuple[AgentBatch, Queue, Queue, List[int]]] = []
        """List of tuples containing all batches and their associated fields:
        batch_list[i][0]: Agent Batch object
        batch_list[i][1]: Queue for sending data to the batch
        batch_list[i][2]: Queue for receiving data from the batch
        batch_list[i][3]: Agent IDs managed by the batch
        """

        # We need at least one agent per batch, and one process for the main simulation
        num_batches = config.multiagent.num_procs-1
        if num_batches > len(agent_id_list):
            num_batches = len(agent_id_list)
        elif num_batches < 1:
            print("Parallel execution requires one process for simulation management.")
            print("Please unset multiprocessing or use > 2 processes.")
            return

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

        ##########################
        #     Run Planning       #
        ##########################

        run_simulation(log_path, config, predictor, scenario, batch_list, agent_id_list, planning_problem_set)


def run_simulation(log_path: str, config: Configuration,
                   predictor: PredictionModule, scenario: Scenario,
                   batch_list: List[Tuple[AgentBatch, Queue, Queue, List[int]]],
                   agent_id_list: List[int], planning_problem_set: PlanningProblemSet):
    """Control a simulation running in multiple processes.

    Computes the predictions, manages the agent patches and the communication.

    :param log_path: Base path for writing the log files to.
    :param config: The configuration.
    :param predictor: Prediction module object used to compute the predictions.
    :param scenario: The scenario to simulate.
    :param batch_list: List of agent batches and associated information, see agent_batch.
    :param agent_id_list: List of IDs of all agents in the simulation.
    :param planning_problem_set: Planning problems of all agents in the simulation.
    """

    # Step simulation as long as some agents have not completed
    init_log(log_path)

    current_timestep = 0

    # Dummy obstacles representing the agents
    dummy_obstacle_list = []

    # Return values of the last agent step
    agent_state_dict = dict()
    for id in agent_id_list:
        agent_state_dict[id] = -1

    # Data from the planners for plotting
    traj_set_list = []
    ref_path_list = []

    running = True
    while running:
        print(f"[Simulation] Simulating timestep {current_timestep}")

        # START TIMER
        step_time_start = time.time()

        ########################
        #    Step Simulation   #
        ########################

        predictions = mh.get_predictions(config, predictor, scenario, current_timestep)

        # Send predictions
        for batch in batch_list:
            batch[1].put(predictions)

        # Plot previous timestep while batches are busy
        # Remove agents that did not exist in the last timestep
        if current_timestep > 0 and (config.debug.show_plots or config.debug.save_plots) \
                and len(batch_list) > 0:
            mh.visualize_multiagent_at_timestep(scenario, planning_problem_set,
                                                list(filter(lambda o: not isnan(o.state_at_time(current_timestep-1).position[0]),
                                                            dummy_obstacle_list)),
                                                current_timestep-1, config, log_path,
                                                traj_set_list=traj_set_list,
                                                ref_path_list=ref_path_list,
                                                predictions=predictions,
                                                plot_window=config.debug.plot_window_dyn)

        # Receive results
        # clear dummy obstacles
        dummy_obstacle_list = []
        for batch in batch_list:
            try:
                dummy_obstacle_list.extend(batch[2].get(block=True, timeout=20))
                agent_state_dict.update(batch[2].get(block=True, timeout=20))
            except Empty:
                print("[Simulation] Timeout while waiting for step results! Exiting")
                return

        # Remove completed workers
        terminated_batch_list = []
        for batch in batch_list:
            if len(list(filter(lambda i: agent_state_dict[i] < 1, batch[3]))) == 0:
                print(f"[Simulation] Terminating batch {batch[3]}...")
                batch[1].put("END", block=True)
                batch[0].join()
                batch[1].close()
                batch[2].close()

                terminated_batch_list.append(batch)

        for batch in terminated_batch_list:
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

        scenario.add_objects(dummy_obstacle_list)

        # STOP TIMER
        sync_time_end = time.time()

        # Receive plotting data
        if (config.debug.show_plots or config.debug.save_plots) and len(batch_list) > 0:
            traj_set_list = []
            ref_path_list = []
            for batch in batch_list:
                try:
                    plotting_values = batch[2].get(block=True, timeout=20)
                    traj_set_list.extend(plotting_values[0])
                    ref_path_list.extend(plotting_values[1])
                except Empty:
                    print("[Simulation] Timeout waiting for plotting data! Exiting")
                    return

        # Write global log
        append_log(log_path, current_timestep, current_timestep * scenario.dt,
                   step_time_end-step_time_start, sync_time_end-sync_time_start,
                   agent_id_list, [agent_state_dict[id] for id in agent_id_list])

        current_timestep += 1
        running = len(batch_list) > 0

    print("[Simulation] Simulation completed.")
    print("[Simulation] Terminating workers...")

    # Workers should already have terminated, otherwise wait for timeouts
    for batch in batch_list:
        batch[0].join()

    if config.debug.gif:
        make_gif(config, scenario, range(0, current_timestep-1), log_path, duration=0.1)
