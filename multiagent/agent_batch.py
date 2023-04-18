import time
import warnings
from multiprocessing import Process, Queue
from queue import Empty

from commonroad_rp.configuration import Configuration

from commonroad.planning.planning_problem import PlanningProblemSet

from multiagent.agent import Agent
from multiagent.multiagent_helpers import visualize_multiagent_at_timestep
from multiagent.multiagent_logging import *


class AgentBatch (Process):

    def __init__(self, agent_id_list: List[int], planning_problem_set: PlanningProblemSet,
                 scenario: Scenario, config: Configuration, log_path: str, mod_path: str,
                 in_queue: Queue, out_queue: Queue):
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
        :param in_queue: Queue the batch receives data from.
        :param out_queue: Queue the batch sends data to.
        """
        super().__init__()

        # Initialize queues
        self.in_queue = in_queue
        self.out_queue = out_queue

        # Initialize agents
        self.agent_id_list = agent_id_list
        self.agent_list = []
        for id in agent_id_list:
            self.agent_list.append(Agent(id, planning_problem_set.find_planning_problem_by_id(id),
                                         scenario, config, os.path.join(log_path, f"{id}"), mod_path))

        # List of all not yet started agents
        self.pending_agent_list = list(filter(lambda a: a.current_timestep > 0, self.agent_list))
        # List of all active agents
        self.running_agent_list = list(filter(lambda a: a.current_timestep == 0, self.agent_list))

        self.current_timestep = 0

    def run(self):
        """Main function of the agent batch.
        Receives predictions from the main simulation, updates the agents,
        runs a planner step, and sends back the dummy obstacles.
        """

        while True:
            # Receive the next predictions
            try:
                predictions = self.in_queue.get(block=True, timeout=10)
            except Empty:
                print("Timeout waiting for new predictions! Exiting.")
                return

            # clear dummy obstacles
            dummy_obstacle_list = list()
            terminated_agent_list = list()

            # Return values of the last agent step
            agent_state_dict = dict()
            for id in self.agent_id_list:
                agent_state_dict[id] = -1

            # START TIMER
            step_time_start = time.time()

            # Step simulation
            for agent in self.running_agent_list:
                print(f"[Batch {self.agent_id_list}] Stepping Agent {agent.id}")

                # Simulate.
                status, dummy_obstacle = agent.step_agent(predictions)

                agent_state_dict[agent.id] = status

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
                    dummy_obstacle_list.append(dummy_obstacle)

            # STOP TIMER
            step_time_end = time.time()

            # Terminate agents
            for agent in terminated_agent_list:
                self.running_agent_list.remove(agent)

            # start pending agents
            for agent in self.pending_agent_list:
                if agent.current_timestep == self.current_timestep + 1:
                    self.running_agent_list.append(agent)
                    dummy_obstacle_list.append(agent.ego_obstacle_list[-1])

                    self.pending_agent_list.remove(agent)

            # Send dummy obstacles to main simulation
            self.out_queue.put(dummy_obstacle_list, block=True)
            self.out_queue.put(agent_state_dict, block=True)

            # Check for active or pending agents
            complete = len(list(filter(lambda v: v < 1, agent_state_dict.values()))) == 0
            if complete:
                # Wait for termination signal from main simulation
                print(f"[Batch {self.agent_id_list}] Completed! Exiting")
                try:
                    self.in_queue.get(block=True, timeout=10)
                except Empty:
                    print(f"[Batch {self.agent_id_list}] Timeout waiting for termination signal.")
                return

            # receive dummy obstacles and outdated agent list
            try:
                dummy_obstacle_list = self.in_queue.get(block=True, timeout=10)
                outdated_agent_id_list = self.in_queue.get(block=True, timeout=10)
            except Empty:
                print("Timeout waiting for agent updates! Exiting")
                return

            # START TIMER
            sync_time_start = time.time()

            # Synchronize agents
            for agent in self.running_agent_list:
                agent.update_scenario(outdated_agent_id_list, dummy_obstacle_list)

            # STOP TIMER
            sync_time_end = time.time()

            self.current_timestep += 1
