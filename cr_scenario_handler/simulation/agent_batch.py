__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from multiprocessing import Process, Queue
from queue import Empty
from typing import Optional

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet

from cr_scenario_handler.simulation.agent import Agent
import cr_scenario_handler.utils.multiagent_helpers as hf
from cr_scenario_handler.utils.multiagent_helpers import AgentStatus
from cr_scenario_handler.utils.multiagent_logging import *


class AgentBatch(Process):

    def __init__(self, agent_id_list: List[int], planning_problem_set: PlanningProblemSet, scenario: Scenario,
                 global_timestep: int, config_planner, config_sim, msg_logger, log_path: str, mod_path: str,
                 in_queue: Optional[Queue] = None, out_queue: Optional[Queue] = None):
        """Batch of agents.

        Manages the Agents in this batch, and communicates dummy obstacles,
        predictions, and plotting data with the main simulation.

        If multiprocessing is enabled, all batches are processed in parallel,
        execution inside one batch is sequential.

        :param agent_id_list: IDs of the agents in this batch.
        :param planning_problem_set: Planning problems of the agents.
        :param scenario: CommonRoad scenario, containing dummy obstacles for all
            running agents from all batches.
        :param config: The configuration.
        :param log_path: Base path of the log files.
        :param mod_path: Path of the working directory of the planners.
        :param in_queue: Queue the batch receives data from (None for serial execution).
        :param out_queue: Queue the batch sends data to (None for serial execution).
        """

        super().__init__()

        self.msg_logger = msg_logger

        # Initialize queues
        self.in_queue = in_queue
        self.out_queue = out_queue

        # Initialize agents
        self.global_timestep = global_timestep

        self.agent_status_dict = dict()
        agent_list = []
        for agent_id in agent_id_list:
            agent = Agent(agent_id, planning_problem_set.find_planning_problem_by_id(agent_id),
                          scenario, config_planner, config_sim, msg_logger)
            agent_list.append(agent)
            self.agent_status_dict[agent_id] = agent.status

        # List of all active agents
        self.latest_starting_time = max([agent.current_timestep for agent in agent_list]) + 1
        self.agent_dict = {timestep: [agent for agent in agent_list if agent.current_timestep == timestep] for timestep
                           in range(self.latest_starting_time)}
        self.running_agent_list = self.agent_dict[self.global_timestep]
        self.agent_history_dict = dict.fromkeys(agent_id_list)
        self.agent_recorded_state_dict = dict.fromkeys(agent_id_list)
        self.agent_input_state_dict = dict.fromkeys(agent_id_list)
        self.agent_ref_path_dict = dict.fromkeys(agent_id_list)
        self.agent_traj_set_dict = dict.fromkeys(agent_id_list)

        self.terminated_agent_list = []
        self.finished = False

    def run(self):
        """ Main function of the agent batch when running in a separate process.

        Receives predictions from the main simulation, updates the agents,
        executes a planner step, and sends back the dummy obstacles and plotting data.

        The messages exchanged with the main simulation are:
            1. predictions: dict received from in_queue:
                    Predictions for all obstacles in the scenario.
            2. dummy_obstacle_list: List[DynamicObstacle] sent to out_queue:
                    New dummy obstacles for all agents in the batch.
            3. agent_state_dict: dict sent to out_queue:
                    Return value of the agent step for every agent ID in the batch.
            If all agents in the batch are terminated:
                4. terminate: Any received from in_queue:
                        Synchronization message triggering the termination of this batch.
            Otherwise:
                4. dummy_obstacle_list: List[DynamicObstacle] received from in_queue:
                        Updated dummy obstacles for all agents in the scenario.
                5. outdated_agent_list: List[int] received from in_queue:
                        List of IDs of agents that have to be updated during synchronization.
                If global plotting is enabled (currently not supported):
                    6. plotting_data: Tuple(List[trajectory bundle], List[reference path])
                                sent to out_queue:
                            Lists of trajectory bundles and reference paths for all agents in the batch.
        """
        while True:
            # Receive the next predictions
            try:
                global_predictions = self.in_queue.get(block=True, timeout=hf.TIMEOUT)
            except Empty:
                self.msg_logger.info(f"Batch {self.name}: Timeout waiting for new predictions!")
                return

            self.step_simulation(global_predictions)

            # Send dummy obstacles to main simulation
            # self.out_queue.put(self.dummy_obstacle_list, block=True)
            # self.out_queue.put(self.agent_state_dict, block=True)
            self.out_queue.put(self.agent_status_dict, block=True)
            self.out_queue.put(self.agent_history_dict, block=True)
            self.out_queue.put(self.agent_recorded_state_dict, block=True)
            self.out_queue.put(self.agent_input_state_dict, block=True)
            self.out_queue.put(self.agent_ref_path_dict, block=True)
            # self.out_queue.put(self.agent_traj_set_dict, block=True)
            # TODO AB HIER!
            # Check for active or pending agents
            if self.finished:
                # Wait for termination signal from main simulation
                self.msg_logger.info(f"Batch {self.name}: Completed!")
                try:
                    self.in_queue.get(block=True, timeout=hf.TIMEOUT)
                except Empty:
                    self.msg_logger.info(f"Batch {self.name}: Timeout waiting for termination signal.")
                return

            # Synchronize agents
            # receive dummy obstacles and outdated agent list
            try:
                args = self.in_queue.get(block=True, timeout=hf.TIMEOUT)
                # self.dummy_obstacle_list = self.in_queue.get(block=True, timeout=hf.TIMEOUT)
                # outdated_agent_id_list = self.in_queue.get(block=True, timeout=hf.TIMEOUT)
            except Empty:
                self.msg_logger.info(f"Batch {self.name}: Timeout waiting for agent updates!")
                return

            self.update_agents(*args)

            self.global_timestep += 1
            # self.out_queue.put(self.global_timestep, block=True)
            if self.global_timestep < self.latest_starting_time:
                self.running_agent_list.extend(self.agent_dict[self.global_timestep])

    # def run_sequential(self, log_path: str, predictor, scenario: Scenario):
    #     """ Main function of the agent batch when running without multiprocessing.
    #
    #     For every time step in the simulation, computes the new predictions,
    #     executes a planning step, synchronizes the agents, and manages
    #     global plotting and logging.
    #
    #     This function contains the combined functionality of Simulation.run_simulation()
    #     and AgentBatch.run() to eliminate communication overhead in a single-process configuration.
    #
    #     :param log_path: Base path for writing the log files to.
    #     :param predictor: Prediction module object used to compute predictions
    #     :param scenario: The scenario to simulate, containing dummy obstacles
    #         for all running agents.
    #     """
    #
    #     init_log(log_path)
    #
    #     while True:
    #         # Calculate the next predictions
    #         predictions = get_predictions(self.config, predictor, scenario, self.current_timestep)
    #
    #         # START TIMER
    #         step_time_start = time.time()
    #
    #         self.step_simulation(predictions)
    #
    #         # STOP TIMER
    #         step_time_end = time.time()
    #
    #         # Check for active or pending agents
    #         if self.complete():
    #             print(f"[Batch {self.agent_id_list}] Completed! Exiting")
    #
    #             if self.config.debug.gif:
    #                 make_gif(scenario, range(0, self.current_timestep-1), log_path, duration=0.1)
    #
    #             return
    #
    #         # Update outdated agent lists
    #         outdated_agent_id_list = list(filter(lambda id: self.agent_state_dict[id] > -1, self.agent_id_list))
    #
    #         # START TIMER
    #         sync_time_start = time.time()
    #
    #         # Update own scenario for predictions and plotting
    #         for id in filter(lambda i: self.agent_state_dict[i] >= 0, self.agent_state_dict.keys()):
    #             # manage agents that are not yet active
    #             with warnings.catch_warnings():
    #                 warnings.filterwarnings("ignore", message=".*not contained in the scenario")
    #                 if scenario.obstacle_by_id(id) is not None:
    #                     scenario.remove_obstacle(scenario.obstacle_by_id(id))
    #
    #         # Plot current frame
    #         if (self.config.debug.show_plots or self.config.debug.save_plots) and \
    #                 len(self.running_agent_list) > 0 and self.config.multiagent.use_multiagent:
    #             visualize_multiagent_at_timestep(scenario, self.planning_problem_set,
    #                                              self.dummy_obstacle_list, self.current_timestep,
    #                                              self.config, log_path,
    #                                              traj_set_list=[a.planner_interface.get_all_traj()
    #                                                             for a in self.running_agent_list],
    #                                              ref_path_list=[a.planner_interface.get_ref_path()
    #                                                             for a in self.running_agent_list],
    #                                              predictions=predictions,
    #                                              plot_window=self.config.debug.plot_window_dyn)
    #         elif (self.config.debug.show_plots or self.config.debug.save_plots) and \
    #                 len(self.running_agent_list) == 1 and not self.config.multiagent.use_multiagent:
    #             curr_planner = self.running_agent_list[0].planner_interface.planner
    #             visualize_planner_at_timestep(scenario=scenario, planning_problem=self.planning_problem_set.
    #                                           find_planning_problem_by_id(self.running_agent_list[0].id),
    #                                           ego=curr_planner.ego_vehicle_history[-1],
    #                                           traj_set=curr_planner.all_traj, optimal_traj=curr_planner.trajectory_pair[0],
    #                                           ref_path=curr_planner.reference_path,
    #                                           timestep=self.current_timestep, config=self.config, predictions=predictions,
    #                                           plot_window=self.config.debug.plot_window_dyn, log_path=log_path)
    #
    #         scenario.add_objects(self.dummy_obstacle_list)
    #
    #         # Synchronize agents
    #         for agent in self.running_agent_list:
    #             agent.update_scenario(outdated_agent_id_list, self.dummy_obstacle_list)
    #
    #         # STOP TIMER
    #         sync_time_end = time.time()
    #
    #         append_log(log_path, self.current_timestep, self.current_timestep * scenario.dt,
    #                    step_time_end - step_time_start, sync_time_end - sync_time_start,
    #                    self.agent_id_list, [self.agent_state_dict[id] for id in self.agent_id_list])
    #
    #         self.current_timestep += 1

    def update_agents(self, scenario: Scenario, colliding_agents: List):
        for agent in self.running_agent_list:
            collision = True if agent.id in colliding_agents else False
            agent.update_agent(scenario, collision)

    def step_simulation(self, global_predictions: dict):
        """Simulate the next timestep.

        Calls the step function of the agents and
        manages starting and terminating agents.

        :param global_predictions: Predictions for all agents in the simulation.
        """

        self.msg_logger.info(f" stepping Batch {self.name}")
        # Step simulation
        for agent in reversed(self.running_agent_list):
            agent.step_agent(global_predictions)
            self.agent_status_dict[agent.id] = agent.status

            if agent.status > hf.AgentStatus.RUNNING:
                self.terminated_agent_list.append(agent)
                msg = "Success" if agent.status == 1 else "Failed"
                with (open(os.path.join(agent.mod_path, "logs", "score_overview.csv"), 'a') as file):
                    line = str(agent.scenario.scenario_id) + ";" + str(agent.current_timestep) + ";" + \
                           str(agent.status) + ";" + msg + "\n"
                    file.write(line)
                self.running_agent_list.remove(agent)
            else:
                self.agent_history_dict[agent.id] = agent.vehicle_history
                self.agent_recorded_state_dict[agent.id] = agent.record_state_list
                self.agent_input_state_dict[agent.id] = agent.record_input_list
                self.agent_ref_path_dict[agent.id] = agent.reference_path
                self.agent_traj_set_dict[agent.id] = agent.traj_set
            #     dummy_obstacle_dict.update(dummy_obstacle)

        self.global_timestep += 1
        if self.global_timestep < self.latest_starting_time:
            self.running_agent_list.extend(self.agent_dict[self.global_timestep])

    def check_completion(self):
        """Check for completion of all agents in this batch."""
        self.finished = all([i > AgentStatus.RUNNING for i in self.agent_status_dict.values()])

