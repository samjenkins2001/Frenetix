import time
import warnings
from math import ceil, isnan
from multiprocessing import Queue
from queue import Empty
from typing import Tuple
import random

# commonroad-io
from commonroad.scenario.state import CustomState

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

from cr_scenario_handler.simulation.agent_batch import AgentBatch
import cr_scenario_handler.utils.multiagent_helpers as mh
from cr_scenario_handler.utils.multiagent_helpers import TIMEOUT
from cr_scenario_handler.utils.multiagent_logging import *

import commonroad_rp.prediction_helpers as ph
from cr_scenario_handler.utils.visualization import visualize_multiagent_at_timestep


class Simulation:

    def __init__(self, config: Configuration, log_path: str, mod_path: str):

        # Configuration
        self.config = config
        self.log_path = log_path
        self.mod_path = mod_path

        # Create and preprocess the scenario, find agent IDs and create planning problems
        self.scenario = None
        self.agent_id_list = []
        self.planning_problem_set = PlanningProblemSet()
        self.scenario_preprocessing()

        # Load prediction framework
        self.predictor = ph.load_prediction(self.scenario, self.config.prediction.mode, self.config)

        # initialize agent batches
        self.batch_list: List[Tuple[AgentBatch, Queue, Queue, List[int]]] = []
        """List of tuples containing all batches and their associated fields:
        batch_list[i][0]: Agent Batch object
        batch_list[i][1]: Queue for sending data to the batch
        batch_list[i][2]: Queue for receiving data from the batch
        batch_list[i][3]: Agent IDs managed by the batch
        """

        self.batch_list = self.create_agent_batches()

        self.current_timestep = 0

        # DynamicObstacles representing the agents in the scenario
        self.dummy_obstacle_list = []

        # Return values of the last agent step
        self.agent_state_dict = dict()
        for id in self.agent_id_list:
            self.agent_state_dict[id] = -1

        # Trajectory bundles and reference paths for plotting
        self.traj_set_list = []
        self.ref_path_list = []

    def scenario_preprocessing(self):
        # Open the commonroad scenario and the planning problems of the
        # original ego vehicles
        self.scenario, ego_planning_problem, original_planning_problem_set = \
            load_scenario_and_planning_problem(self.config)

        # Select obstacles IDs to use as agents
        self.agent_id_list = self.select_agents()

        # Create PlanningProblems for additional agents
        initial_obstacle_list, self.planning_problem_set = \
            self.create_planning_problems_for_obstacles()

        # Add obstacles for original agents
        for problem in original_planning_problem_set.planning_problem_dict.values():
            dummy_obstacle, planning_problem = self.create_obstacle_for_planning_problem(problem)
            self.agent_id_list.append(planning_problem.planning_problem_id)
            initial_obstacle_list.append(dummy_obstacle)
            self.scenario.add_objects(dummy_obstacle)
            self.planning_problem_set.add_planning_problem(planning_problem)

        # Remove pending agents from the scenario
        for obs in initial_obstacle_list:
            if obs.initial_state.time_step > 0:
                self.scenario.remove_obstacle(obs)

    def run_simulation(self):
        """ Based on commonroad_rp/run_planner.py
        Set up the configuration and the simulation.
        Creates and manages the agents batches and their communication.
        """

        # start agent batches
        if len(self.batch_list) == 1:
            # If we have only one batch, run sequential simulation
            self.batch_list[0][0].run_sequential(self.log_path, self.predictor, self.scenario)
        else:
            # start parallel batches
            for batch in self.batch_list:
                batch[0].start()

            # run parallel simulation
            self.run_parallel_simulation()

    def select_agents(self):
        """Select the dynamic obstacles that should be simulated as agents.

        :return: A List of obstacle IDs that should be used as agents
        """

        # Find all dynamic obstacles in the scenario
        all_obs_ids = mh.get_all_obstacle_ids(self.scenario)

        if self.config.multiagent.use_specific_agents:
            # Agents were selected by the user
            agent_id_list = self.config.multiagent.agent_ids
            for agent_ids_check in agent_id_list:
                if agent_ids_check not in all_obs_ids:
                    raise ValueError("Selected Obstacle IDs not existent in Scenario!\n"
                                     "Check selected 'agent_ids' in config!")
        else:
            agent_id_list = all_obs_ids

        if self.config.multiagent.number_of_agents < len(agent_id_list):
            if self.config.multiagent.select_agents_randomly:
                # Choose agents randomly
                agent_id_list = random.sample(agent_id_list, self.config.multiagent.number_of_agents)
            else:
                # Choose the first few obstacles in the scenario
                agent_id_list = agent_id_list[:self.config.multiagent.number_of_agents]

        return agent_id_list

    def create_planning_problems_for_obstacles(self):
        obstacle_list = []
        planning_problem_set = PlanningProblemSet()

        for id in self.agent_id_list:
            obstacle = self.scenario.obstacle_by_id(id)
            obstacle_list.append(obstacle)
            initial_state = obstacle.initial_state
            if not hasattr(initial_state, 'acceleration'):
                initial_state.acceleration = 0.

            # create planning problem
            # Allow deviation from final state:
            #     time: +/- 5 time steps
            #     position: Circle with 3m diameter around final state
            #     velocity: +/- 2 m/s
            #     orientation: +/- 20°
            final_state = obstacle.prediction.trajectory.final_state
            goal_state = CustomState(time_step=Interval(final_state.time_step - 5, final_state.time_step + 5),
                                     position=Circle(1.5, final_state.position),
                                     velocity=Interval(final_state.velocity - 2, final_state.velocity + 2),
                                     orientation=AngleInterval(final_state.orientation - 0.349,
                                                               final_state.orientation + 0.349))

            problem = PlanningProblem(id, initial_state, GoalRegion(list([goal_state])))
            planning_problem_set.add_planning_problem(problem)

        return obstacle_list, planning_problem_set

    def create_agent_batches(self):

        if not self.config.multiagent.multiprocessing \
                or len(self.agent_id_list) < 2 \
                or self.config.multiagent.num_procs < 3:
            # Multiprocessing disabled or useless, run single process
            agent_batch = AgentBatch(self.agent_id_list, self.planning_problem_set, self.scenario,
                                     self.config, self.log_path, self.mod_path,
                                     None, None)

            return [(agent_batch, None, None, self.agent_id_list)]

        else:
            batch_list = []

            # We need at least one agent per batch, and one process for the main simulation
            num_batches = self.config.multiagent.num_procs-1
            if num_batches > len(self.agent_id_list):
                num_batches = len(self.agent_id_list)
            elif num_batches < 1:
                print("Parallel execution requires one process for simulation management.")
                print("Please unset multiprocessing or use > 2 processes.")
                return

            batch_size = ceil(len(self.agent_id_list) / num_batches)

            for i in range(num_batches):
                inqueue = Queue()
                outqueue = Queue()

                batch_list.append((AgentBatch(self.agent_id_list[i * batch_size:(i + 1) * batch_size],
                                              self.planning_problem_set, self.scenario,
                                              self.config, self.log_path, self.mod_path,
                                              outqueue, inqueue),
                                   outqueue,
                                   inqueue,
                                   self.agent_id_list[i * batch_size:(i + 1) * batch_size]))

            return batch_list

    def create_obstacle_for_planning_problem(self, planning_problem: PlanningProblem):
        id = planning_problem.planning_problem_id
        if not hasattr(planning_problem.initial_state, 'acceleration'):
            planning_problem.initial_state.acceleration = 0.

        # add dummy obstacle for original ego vehicle
        vehicle_params = self.config.vehicle
        dummy_obstacle = StaticObstacle(
            id,
            ObstacleType.CAR,
            Rectangle(length=vehicle_params.length, width=vehicle_params.width),
            planning_problem.initial_state)

        return dummy_obstacle, planning_problem

    def run_parallel_simulation(self):
        """Control a simulation running in multiple processes.

        Computes the predictions, manages the agent patches and the communication.
        """

        init_log(self.log_path)

        running = True
        while running:
            running = self.step_parallel_simulation()

        print("[Simulation] Simulation completed.")

        # Workers should already have terminated, otherwise wait for timeouts
        print("[Simulation] Terminating workers...")
        for batch in self.batch_list:
            batch[0].join()

        if self.config.debug.gif:
            make_gif(self.config, self.scenario, range(0, self.current_timestep-1), self.log_path, duration=0.1)

    def step_parallel_simulation(self):

        print(f"[Simulation] Simulating timestep {self.current_timestep}")

        # START TIMER
        step_time_start = time.time()

        # Calculate new predictions
        predictions = mh.get_predictions(self.config, self.predictor, self.scenario, self.current_timestep)

        # Send predictions to agent batches
        for batch in self.batch_list:
            batch[1].put(predictions)

        # Plot previous timestep while batches are busy
        # Remove agents that did not exist in the last timestep
        if self.current_timestep > 0 and (self.config.debug.show_plots or self.config.debug.save_plots) \
                and len(self.batch_list) > 0:
            visualize_multiagent_at_timestep(self.scenario, self.planning_problem_set,
                                             list(filter(lambda o: not isnan(
                                                 o.state_at_time(self.current_timestep - 1).position[0]),
                                                         self.dummy_obstacle_list)),
                                             self.current_timestep - 1, self.config, self.log_path,
                                             traj_set_list=self.traj_set_list,
                                             ref_path_list=self.ref_path_list,
                                             predictions=predictions,
                                             plot_window=self.config.debug.plot_window_dyn)

        # Receive simulation step results
        self.dummy_obstacle_list = []
        for batch in self.batch_list:
            try:
                self.dummy_obstacle_list.extend(batch[2].get(block=True, timeout=TIMEOUT))
                self.agent_state_dict.update(batch[2].get(block=True, timeout=TIMEOUT))
            except Empty:
                print("[Simulation] Timeout while waiting for step results! Exiting")
                return

        # Remove completed workers
        terminated_batch_list = []
        for batch in self.batch_list:
            if len(list(filter(lambda i: self.agent_state_dict[i] < 1, batch[3]))) == 0:
                print(f"[Simulation] Terminating batch {batch[3]}...")
                batch[1].put("END", block=True)
                batch[0].join()
                batch[1].close()
                batch[2].close()

                terminated_batch_list.append(batch)

        for batch in terminated_batch_list:
            self.batch_list.remove(batch)

        # STOP TIMER
        step_time_end = time.time()

        # Update list of agents that were changed in this simulation step
        outdated_agent_list = list(filter(lambda id: self.agent_state_dict[id] > -1, self.agent_id_list))

        # START TIMER
        sync_time_start = time.time()

        # Send dummy obstacles
        for batch in self.batch_list:
            batch[1].put(self.dummy_obstacle_list, block=True)
            batch[1].put(outdated_agent_list, block=True)

        # Update own scenario for predictions and plotting
        for id in filter(lambda i: self.agent_state_dict[i] >= 0, self.agent_state_dict.keys()):
            # manage agents that are not yet active
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not contained in the scenario")
                if self.scenario.obstacle_by_id(id) is not None:
                    self.scenario.remove_obstacle(self.scenario.obstacle_by_id(id))

        self.scenario.add_objects(self.dummy_obstacle_list)

        # STOP TIMER
        sync_time_end = time.time()

        # Receive plotting data
        if (self.config.debug.show_plots or self.config.debug.save_plots) and len(self.batch_list) > 0:
            self.traj_set_list = []
            self.ref_path_list = []
            for batch in self.batch_list:
                try:
                    plotting_values = batch[2].get(block=True, timeout=TIMEOUT)
                    self.traj_set_list.extend(plotting_values[0])
                    self.ref_path_list.extend(plotting_values[1])
                except Empty:
                    print("[Simulation] Timeout waiting for plotting data! Exiting")
                    return

        # Write global log
        append_log(self.log_path, self.current_timestep, self.current_timestep * self.scenario.dt,
                   step_time_end - step_time_start, sync_time_end - sync_time_start,
                   self.agent_id_list, [self.agent_state_dict[id] for id in self.agent_id_list])

        self.current_timestep += 1

        return len(self.batch_list) > 0