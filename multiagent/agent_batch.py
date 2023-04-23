import time
import warnings
from multiprocessing import Process, Queue
from queue import Empty
from typing import Union, Optional

from commonroad_prediction.prediction_module import PredictionModule

from commonroad_rp.configuration import Configuration

from commonroad.planning.planning_problem import PlanningProblemSet

from multiagent.agent import Agent
from multiagent.multiagent_helpers import get_predictions, visualize_multiagent_at_timestep, make_gif
from multiagent.multiagent_logging import *


class AgentBatch (Process):

    def __init__(self, agent_id_list: List[int], planning_problem_set: PlanningProblemSet,
                 scenario: Scenario, config: Configuration, log_path: str, mod_path: str,
                 in_queue: Optional[Queue], out_queue: Optional[Queue]):
        """Batch of agents.

        If multiprocessing is enabled, all batches are processed in parallel,
        execution inside one batch is sequential.

        Manages the Agents in this batch, and communicates dummy obstacles and
        predictions with the main simulation at run_multiagent.py.

        :param agent_id_list: IDs of the agents in this batch.
        :param planning_problem_set: Planning problems of the agents.
        :param scenario: CommonRoad scenario, containing dummy obstacles for all
                         running agents from all batches.
        :param config: The configuration.
        :param log_path: Base path of the log files.
        :param mod_path: Path of the working directory of the planners
                         (containing planner configuration)
        :param in_queue: Queue the batch receives data from (None for serial execution).
        :param out_queue: Queue the batch sends data to (None for serial execution).
        """
        super().__init__()

        self.config = config

        # Initialize queues
        if config.multiagent.multiprocessing and \
            (in_queue is None or out_queue is None):
            print(f"[Batch {agent_id_list}] Communication queues may not be None for parallel execution! Exiting")
            return
        self.in_queue = in_queue
        self.out_queue = out_queue

        # Initialize agents
        self.agent_id_list = agent_id_list
        self.planning_problem_set = planning_problem_set
        self.agent_list = []
        for id in agent_id_list:
            self.agent_list.append(Agent(id, planning_problem_set.find_planning_problem_by_id(id),
                                         scenario, config, os.path.join(log_path, f"{id}"), mod_path))

        # List of all not yet started agents
        self.pending_agent_list = list(filter(lambda a: a.current_timestep > 0, self.agent_list))
        # List of all active agents
        self.running_agent_list = list(filter(lambda a: a.current_timestep == 0, self.agent_list))

        # List of all dummy obstacles
        self.dummy_obstacle_list = []
        # Exit codes of the agent steps
        self.agent_state_dict = dict()
        for id in self.agent_id_list:
            self.agent_state_dict[id] = -1

        self.current_timestep = 0

    def step_simulation(self, predictions: dict):
        """Simulate the next timestep.

        Calls the step function of the agents and
        manages starting and terminating agents.

        :param predictions: Predictions for all agents in the simulation.
        """

        # clear dummy obstacles
        self.dummy_obstacle_list = []
        terminated_agent_list = []

        # Step simulation
        for agent in self.running_agent_list:
            print(f"[Batch {self.agent_id_list}] Stepping Agent {agent.id}")

            # Simulate.
            status, dummy_obstacle = agent.step_agent(predictions)

            self.agent_state_dict[agent.id] = status

            msg = ""
            if status > 0:
                if status == 1:
                    msg = "Completed."
                elif status == 2:
                    msg = "Failed to find valid trajectory."
                elif status == 3:
                    msg = "Collision detected."

                print(f"[Batch {self.agent_id_list}] Agent {agent.id} terminated: {msg}")
                # Terminate all agents simultaneously
                terminated_agent_list.append(agent)
            else:
                # save dummy obstacle
                self.dummy_obstacle_list.append(dummy_obstacle)

        # Terminate agents
        for agent in terminated_agent_list:
            self.running_agent_list.remove(agent)

        # start pending agents
        for agent in self.pending_agent_list:
            if agent.current_timestep == self.current_timestep + 1:
                self.running_agent_list.append(agent)
                self.dummy_obstacle_list.append(agent.ego_obstacle_list[-1])

                self.pending_agent_list.remove(agent)

    def complete(self):
        """Check for completion of all agents in this batch."""
        return len(list(filter(lambda v: v < 1, self.agent_state_dict.values()))) == 0

    def run(self):
        """Main function of the agent batch.
        Receives predictions from the main simulation, updates the agents,
        runs a planner step, and sends back the dummy obstacles.
        """

        while True:
            # Receive the next predictions
            try:
                predictions = self.in_queue.get(block=True, timeout=20)
            except Empty:
                print(f"[Batch {self.agent_id_list}] Timeout waiting for new predictions! Exiting.")
                return

            self.step_simulation(predictions)

            # Send dummy obstacles to main simulation
            self.out_queue.put(self.dummy_obstacle_list, block=True)
            self.out_queue.put(self.agent_state_dict, block=True)

            # Check for active or pending agents
            if self.complete():
                # Wait for termination signal from main simulation
                print(f"[Batch {self.agent_id_list}] Completed! Exiting")
                try:
                    self.in_queue.get(block=True, timeout=20)
                except Empty:
                    print(f"[Batch {self.agent_id_list}] Timeout waiting for termination signal.")
                return

            # receive dummy obstacles and outdated agent list
            try:
                self.dummy_obstacle_list = self.in_queue.get(block=True, timeout=20)
                outdated_agent_id_list = self.in_queue.get(block=True, timeout=20)
            except Empty:
                print("[Batch {self.agent_id_list}] Timeout waiting for agent updates! Exiting")
                return

            # Synchronize agents
            for agent in self.running_agent_list:
                agent.update_scenario(outdated_agent_id_list, self.dummy_obstacle_list)

            # Send data for global plotting
            if self.config.debug.show_plots or self.config.debug.save_plots:
                self.out_queue.put(([a.planner.get_all_traj() for a in self.running_agent_list],
                                    [a.planner.get_ref_path() for a in self.running_agent_list]))

            self.current_timestep += 1

    def run_sequential(self, log_path: str, predictor: PredictionModule, scenario: Scenario):
        """Main function of the agent batch.
        Receives predictions from the main simulation, updates the agents,
        runs a planner step, and sends back the dummy obstacles.

        Version without agent-level multiprocessing.

        :param log_path: Base path for writing the log files to.
        :param predictor: Prediction module object used to compute predictions
        :param scenario: The scenario to simulate, containing dummy obstacles
                         for all running agents.
        """

        init_log(log_path)

        while True:
            # Calculate the next predictions
            predictions = get_predictions(self.config, predictor, scenario, self.current_timestep)

            # START TIMER
            step_time_start = time.time()

            self.step_simulation(predictions)

            # STOP TIMER
            step_time_end = time.time()

            # Check for active or pending agents
            if self.complete():
                print(f"[Batch {self.agent_id_list}] Completed! Exiting")

                if self.config.debug.gif:
                    make_gif(scenario, range(0, self.current_timestep-1), log_path, duration=0.1)

                return

            # Update outdated agent lists
            outdated_agent_id_list = list(filter(lambda id: self.agent_state_dict[id] > -1, self.agent_id_list))

            # START TIMER
            sync_time_start = time.time()

            # Update own scenario for predictions and plotting
            for id in filter(lambda i: self.agent_state_dict[i] >= 0, self.agent_state_dict.keys()):
                # manage agents that are not yet active
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*not contained in the scenario")
                    if scenario.obstacle_by_id(id) is not None:
                        scenario.remove_obstacle(scenario.obstacle_by_id(id))

            # Plot current frame
            if (self.config.debug.show_plots or self.config.debug.save_plots) and len(self.running_agent_list) > 0:
                visualize_multiagent_at_timestep(scenario, self.planning_problem_set,
                                                 self.dummy_obstacle_list, self.current_timestep, self.config, log_path,
                                                 traj_set_list=[a.planner.get_all_traj() for a in self.running_agent_list],
                                                 ref_path_list=[a.planner.get_ref_path() for a in self.running_agent_list],
                                                 predictions=predictions,
                                                 plot_window=self.config.debug.plot_window_dyn)

            scenario.add_objects(self.dummy_obstacle_list)

            # Synchronize agents
            for agent in self.running_agent_list:
                agent.update_scenario(outdated_agent_id_list, self.dummy_obstacle_list)

            # STOP TIMER
            sync_time_end = time.time()

            append_log(log_path, self.current_timestep, self.current_timestep * scenario.dt,
                       step_time_end - step_time_start, sync_time_end - sync_time_start,
                       self.agent_id_list, [self.agent_state_dict[id] for id in self.agent_id_list])

            self.current_timestep += 1