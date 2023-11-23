__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from multiprocessing import Process, Queue
from queue import Empty
from typing import Optional
import time

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
        :param planning_problem_set:  Planning problems of the agents.
        :param scenario: CommonRoad scenario, containing dummy obstacles for all
            running agents from all batches.
        :param global_timestep: initial simulation timestep
        :param config_planner: configuration of planner
        :param config_sim: configuration of simulation
        :param msg_logger: logger
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

        # Initialize batch
        self.global_timestep = global_timestep
        self.agent_list = []
        for agent_id in agent_id_list:
            # Initialize Agents
            agent = Agent(agent_id, planning_problem_set.find_planning_problem_by_id(agent_id),
                          scenario, config_planner, config_sim, msg_logger)
            self.agent_list.append(agent)

        # initialize communication dict
        self.out_queue_dict = dict.fromkeys(agent_id_list, {})

        self.latest_starting_time = max([agent.current_timestep for agent in self.agent_list])

        # dict of all agents with corresponding starting times
        self.agent_dict = {timestep: [agent for agent in self.agent_list if agent.current_timestep == timestep] for
                           timestep in range(self.latest_starting_time+1)}
        # list of all active agents
        self.running_agent_list = []
        # list of all finished agents
        self.terminated_agent_list = []
        self.finished = False

        self.process_times = dict()

    def run(self):
        """ Main function of the agent batch when running in a separate process.

        Receives necessary information from the main simulation, and performs one simulation step in the batch,
        sends back agent updates and the batch status


        The messages exchanged with the main simulation are:
        in_queue (args):
            -input for self.step_simulation: [scenario, global_timestep, global_predictions, colliding_agents]
            - if all agents are terminated: any synchronization message triggering the termination of this batch.
        out_queue:
            - out_queue_dict: current agent update (see self._update_batch())
            - batch status

        """
        while True:
            # Receive the next predictions
            # Synchronize agents
            # receive dummy obstacles and outdated agent list
            start_time = time.perf_counter()
            self.process_times[self.global_timestep+1] = dict()

            try:
                args = self.in_queue.get(block=True, timeout=hf.TIMEOUT)
            except Empty:
                self.msg_logger.error(f"Batch {self.name}: Timeout waiting for "
                                      f"{'simulation' if self.finished else 'agent'} updates!")
                return

            if self.finished:
                # if batch finished, postprocess agents (currently only make_gif())
                self.msg_logger.info(f"Batch {self.name}: Simulation of the batch finished!")
                for agent in self.terminated_agent_list:
                    agent.make_gif()
                return

            else:
                # simulate next step

                self.step_simulation(*args)

                # send agent updates to simulation
                self.out_queue.put(self.out_queue_dict)

                self.process_times[self.global_timestep].update({"single_process_run": time.perf_counter() - start_time})
                # send batch status to simulation
                proc_time = self.process_times if self.finished else None
                self.out_queue.put([self.finished, proc_time])

            # TODO: Sending trajectory bundles between processes is currently unsupported.

    def step_simulation(self, scenario, global_timestep, global_predictions, colliding_agents):
        """Simulate the next timestep.

        Adds later starting agents to running list,
        updates agents with current scenario, prediction and colliding agent-IDs and
        calls the step function of the agents.
        After each step, the status of the agents within the batch is updated and the batch checks for its completion.

        :param scenario: current valid (global) scenario representation
        :param global_timestep: current global timestep
        :param global_predictions: prediction dict with all obstacles within the scenario
        :param colliding_agents: list with IDs of agents that collided in the prev. timestep
        """

        self.msg_logger.debug(f"Stepping Batch {self.name}")
        step_time = time.perf_counter()
        # update batch timestep
        self.global_timestep = global_timestep
        self.process_times[self.global_timestep] = dict()

        # add agents if they enter the scenario
        if self.global_timestep <= self.latest_starting_time:
            self.running_agent_list.extend(self.agent_dict[self.global_timestep])

        # update agents
        agent_update_time = time.perf_counter()
        self._update_agents(scenario, global_predictions, colliding_agents)
        agent_update_time = time.perf_counter()-agent_update_time

        # step simulation
        single_step_time = time.perf_counter()
        self._step_agents(global_timestep)
        single_step_time = time.perf_counter() - single_step_time

        # update batch
        batch_update = time.perf_counter()
        self._update_batch()
        # check for batch completion
        self._check_completion()
        batch_update = time.perf_counter() - batch_update
        self.process_times[self.global_timestep].update({"complete_simulation_step": time.perf_counter() - step_time,
                                                         "agent_update": agent_update_time,
                                                         "step_duration": single_step_time,
                                                         "batch_update": batch_update})

    def _update_agents(self, scenario: Scenario,global_predictions: dict,  colliding_agents: List):
        for agent in self.running_agent_list:
            # update agent if he collided and update predictions and scenario
            collision = True if agent.id in colliding_agents else False
            agent.update_agent(scenario, global_predictions, collision)

    def _step_agents(self, global_timestep):
        for agent in self.running_agent_list:
            # plan one step in each agent
            agent.step_agent(global_timestep)

    def _update_batch(self):
        """
        update agent lists and prepare dict to send to simulation
        Current agent update:
            - Agent status
            - Agent collision objects for the global collision check
            - Agent vehicle history for visualization
        """
        for agent in reversed(self.running_agent_list):
            if agent.status > hf.AgentStatus.RUNNING:
                self.terminated_agent_list.append(agent)
                self.running_agent_list.remove(agent)
                msg = "Success" if agent.status == 1 else "Failed"
                with (open(os.path.join(agent.mod_path, "logs", "score_overview.csv"), 'a') as file):
                    line = str(agent.scenario.scenario_id) + ";" + str(agent.current_timestep) + ";" + \
                           str(agent.status) + ";" + msg + "\n"
                    file.write(line)

            self.out_queue_dict[agent.id] = {"status": agent.agent_state,
                                             "collision_objects": [agent.collision_objects[-1]],
                                             "vehicle_history": [agent.vehicle_history[-1]],
                                             "record_state_list": [agent.record_state_list[-1]],
                                             "traj_set": agent.traj_set
                                             }

    def _check_completion(self):
        """
        check for completion of all agents in this batch.
        """
        self.finished = all([i.status > AgentStatus.RUNNING for i in self.agent_list])
        return
