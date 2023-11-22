__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import copy
import random
import time
from math import ceil
from multiprocessing import Queue
from queue import Empty
from typing import List

import psutil
from commonroad.common.util import AngleInterval
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle, Circle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import ObstacleType, ObstacleRole, DynamicObstacle
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory

# commonroad-dc
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_dc.collision.trajectory_queries import trajectory_queries

# cr-scenario-handler
import cr_scenario_handler.utils.general as general
import cr_scenario_handler.utils.multiagent_helpers as hf
import cr_scenario_handler.utils.multiagent_logging as multi_agent_log
import cr_scenario_handler.utils.prediction_helpers as ph
import cr_scenario_handler.utils.visualization as visu
from cr_scenario_handler.utils.visualization import visualize_multiagent_scenario_at_timestep
from cr_scenario_handler.simulation.agent_batch import AgentBatch
from cr_scenario_handler.utils.multiagent_helpers import TIMEOUT, AgentStatus

# msg_logger = logging.getLogger("Simulation_logger")


class Simulation:

    def __init__(self, config_sim, config_planner):
        """ Main class for running a planner on a scenario.

        Manages the global configuration, creates and manages agent batches,
        handles global communication and coordination during parallel simulations,
        and does simulation-level logging and plotting.

        :param config_sim: simulation configuration
        :param config_planner: planner configuration
        :param config_eval: simulation evaluation configuration
        """

        # Configuration
        init_time_start = time.perf_counter()

        self.config = config_sim
        self.config_simulation = config_sim.simulation
        self.config_visu = config_sim.visualization
        self.config_eval = None

        self.mod_path = self.config_simulation.mod_path
        self.log_path = self.config_simulation.log_path

        self._multiproc = self.config_simulation.multiprocessing
        # use specified number of processors, else use all physical cores (-1 bc of the main process)
        self._num_procs = self.config_simulation.num_procs if not self.config_simulation.num_procs == -1 else (
                psutil.cpu_count(logical=False) - 1)

        # initialize simulation logger
        self.msg_logger = multi_agent_log.logger_initialization(config_sim, self.log_path, "Simulation_logger")
        self.msg_logger.critical("Simulating Scenario: " + self.log_path.split("/")[-1])
        # TODO Simulation-Logging
        # multi_agent_log.init_log(self.log_path)

        # Create and preprocess the scenario, create planning problems and find agent-IDs
        self.global_timestep = 0
        scenario, self.planning_problem_set = self._open_scenario()
        if self.config_simulation.use_multiagent:
            # create additional planning-problems for multiagent simulation
            # by converting dynamic obstacles into planning problems
            multi_agent_planning_problems = self._create_multiagent_planning_problems(scenario)
            # update planning-problem-set with new planning problems
            [self.planning_problem_set.add_planning_problem(agent_problem) for agent_problem in
             multi_agent_planning_problems]
        # get all agents IDs
        self.agent_id_list = list(self.planning_problem_set.planning_problem_dict.keys())

        self.scenario = self._preprocess_scenario(scenario)

        # additional external Modules
        self._predictor = None
        self._occlusion = None
        self._reach_set = None

        # load additional external modules
        self._load_external_modules()

        # create list with agent batches for parallel batch-wise processing
        self.batch_list = self._create_agent_batches(config_planner)
        # store all agents in one list
        self.agents = []
        [self.agents.extend(batch.agent_list) for batch in self.batch_list]

        # prepare global collision checker
        self._cc_dyn = None
        self._cc_stat = None
        self._set_collision_check()

        # initialize prediction horizon and horizon
        self.global_predictions = None
        # get prediction horizon if specified in planner-configuration, else use 2 seconds by default
        self.prediction_horizon = config_planner.planning.planning_horizon if (hasattr(config_planner, "planning")
                                                                               and hasattr(config_planner.planning,
                                                                                           "planning_horizon")) else 2

        self.process_times = dict()
        self.process_times["simulation_steps"] = dict()
        self.process_times["initialization"] = time.perf_counter() - init_time_start

    def _open_scenario(self):
        """
        opens the scenario specified
        :return: the scenario and the original planning-problem set
        """
        scenario, _, original_planning_problem_set = general.load_scenario_and_planning_problem(self.config)
        for planning_problem in original_planning_problem_set.planning_problem_dict.values():
            if not hasattr(planning_problem.initial_state, 'acceleration'):
                planning_problem.initial_state.acceleration = 0.
        return scenario, original_planning_problem_set

    def _create_multiagent_planning_problems(self, scenario):
        """Modify a commonroad scenario to prepare it for the simulation.

        Reads the scenario and planning problem from the configuration,
        selects the agents to be simulated, and creates missing planning problems

        :param scenario: cr-scenario to be simulated
        :return: a list with new additional planning problems
        """
        # select additional agents
        multi_agent_id_list = self._select_additional_agents(scenario)

        # Create planning-problems for additional agents
        multi_agent_planning_problems = self._create_planning_problems_for_agent_obstacles(multi_agent_id_list,
                                                                                           scenario)
        return multi_agent_planning_problems

    def _select_additional_agents(self, scenario):
        """ Selects the dynamic obstacles that should be simulated as agents
        according to the multiagent configuration.

        :return: A List of obstacle IDs that should be used as agents
        """
        # Find all dynamic obstacles in the scenario with a supported obstacle type
        allowed_types = [ObstacleType.CAR,
                         # ObstacleType.TRUCK,
                         # ObstacleType.BUS
                         ]
        allowed_roles = [ObstacleRole.DYNAMIC]
        allowed_id_list = [obs.obstacle_id for obs in scenario.obstacles if obs.obstacle_type in allowed_types and
                           obs.obstacle_role in allowed_roles]

        if self.config_simulation.use_specific_agents:
            # Agents were selected by the user
            obstacle_agent_id_list = self.config_simulation.agent_ids
            for agent_id in obstacle_agent_id_list:
                if agent_id not in allowed_id_list:
                    raise ValueError(f"Selected Obstacle ID {agent_id} not existent in Scenario,"
                                     f"or of unsupported ObstacleType!\n"
                                     "Check selected 'agent_ids' in config!")
            return obstacle_agent_id_list

        if -1 < self.config_simulation.number_of_agents < len(allowed_id_list):
            if self.config_simulation.select_agents_randomly:
                # Choose agents randomly
                obstacle_agent_id_list = list(random.sample(allowed_id_list, self.config_simulation.number_of_agents))
            else:
                # Choose the first few obstacles in the scenario
                obstacle_agent_id_list = allowed_id_list[:self.config_simulation.number_of_agents]
        else:
            # Use all obstacles as agents
            obstacle_agent_id_list = allowed_id_list

        return obstacle_agent_id_list

    def _create_planning_problems_for_agent_obstacles(self, multi_agent_id_list, scenario):

        """ Creates the missing planning problems for agents that should be
        created from dynamic obstacles.

        The goal state is defined as a small area around the final state of the
        trajectory of the dynamic obstacle.
        The allowed deviations from this state are:
            time: +/- 20 time steps from final time step
            position: Circle with 3m diameter around final state
            velocity: +/- 2 m/s from final state
            orientation: +/- 20° from final state

        :return: planning_problem_list with all created problems
        """

        planning_problem_list = []
        for agent_id in multi_agent_id_list:
            if not id in self.planning_problem_set.planning_problem_dict.keys():
                # get dynamic obstacle of selected agent
                obstacle = scenario.obstacle_by_id(agent_id)
                # define initial state for planning problem
                initial_state = obstacle.initial_state
                if not hasattr(initial_state, 'acceleration'):
                    initial_state.acceleration = 0.
                # define goal state for planning problem
                final_state = obstacle.prediction.trajectory.final_state
                goal_state = CustomState(time_step=Interval(final_state.time_step - 20, final_state.time_step + 20),
                                         position=Circle(1.5, final_state.position),
                                         velocity=Interval(final_state.velocity - 2, final_state.velocity + 2),
                                         orientation=AngleInterval(final_state.orientation - 0.349,
                                                                   final_state.orientation + 0.349))
                # create planning problem
                problem = PlanningProblem(agent_id, initial_state, GoalRegion(list([goal_state])))
                planning_problem_list.append(problem)

        return planning_problem_list

    def _preprocess_scenario(self, scenario):
        """ Add dummy obstacles for all planning_problems to the scenario
        :param scenario: simulated scenario to be processed
        :return: the processed scenario
        """
        # Add obstacles for planning problems
        for problem_id, planning_problem in self.planning_problem_set.planning_problem_dict.items():
            # create dummy obstacle
            dummy_obstacle = self._create_obstacle_for_planning_problem(planning_problem, scenario)
            # remove existing obstacles of the agents
            scenario = hf.scenario_without_obstacle_id(scenario, [problem_id])
            # add dummys to scenario
            scenario.add_objects(dummy_obstacle)
        return scenario

    def _create_obstacle_for_planning_problem(self, planning_problem: PlanningProblem, scenario):
        """ Creates a dummy obstacle from a given planning problem.

        Extends the initial state of the planning problem to include all necessary values,
        and creates a DynamicObstacle from the initial state of the planning problem
        and the vehicle configuration.
        The prediction of the new obstacle contains only its current state.

        :param planning_problem: planning problem to create a dummy obstacle for
        :return: dummy_obstacle at initial state of the planning problem
        """
        # get vehicle id
        problem_id = planning_problem.planning_problem_id
        if not hasattr(planning_problem.initial_state, 'acceleration'):
            planning_problem.initial_state.acceleration = 0.

        # create dummy obstacle from the planning problem
        vehicle_params = self.config.vehicle
        shape = Rectangle(length=vehicle_params.length, width=vehicle_params.width)
        ini_state = planning_problem.initial_state
        lanelet_assigned= {0: set(scenario.lanelet_network.find_lanelet_by_position([ini_state.position])[0])}
        dummy_obstacle = DynamicObstacle(
            problem_id,
            ObstacleType.CAR,
            shape,
            ini_state,
            TrajectoryPrediction(Trajectory(planning_problem.initial_state.time_step,
                                            [planning_problem.initial_state]),
                                             shape, center_lanelet_assignment=lanelet_assigned)
        )
        # otherwise prediction of this obstacle not available not used
        dummy_obstacle.prediction.final_time_step += 1
        dummy_obstacle.prediction.initial_time_step += 1
        return dummy_obstacle

    def _load_external_modules(self):
        """
        load all external modules
        """
        # load prediction framework
        # TODO include gt if not multiagent
        self._predictor = ph.load_prediction(self.scenario, self.config.prediction.mode)

        """---------------------------------------"""
        # TODO include CR-Reach, Occlusion-Module,Sensormodel falls es verschiedene gibt
        # load reach set
        # if 'responsibility' in config.cost.cost_weights and config.cost.cost_weights['responsibility'] > 0:
        #     self._reach_set = ph.load_reachset(self.scenario, self.config, mod_path)

        """----------------------------------------"""

    def _create_agent_batches(self, config_planner):
        """ Initialize the agent batches and set up the communication queues.

        Reads the configuration to determine the number of agent batches to create,
        creates the batches, and establishes the communication queues.

        :return: batch_list: List of batches with agent objects used for planning
        """
        batch_list: List[AgentBatch] = []
        if not self._multiproc or self._num_procs < 3 \
                or len(self.agent_id_list) < 2:

            # Multiprocessing disabled or useless, run single process
            batch_list.append(AgentBatch(self.agent_id_list, self.planning_problem_set, self.scenario,
                                         self.global_timestep, config_planner, self.config,
                                         self.msg_logger, self.log_path, self.mod_path))

        else:
            # We need at least one agent per batch, and one process for the main simulation

            chunk_size = ceil(len(self.agent_id_list) / (self._num_procs))
            chunks = [self.agent_id_list[ii * chunk_size:
                                         min(len(self.agent_id_list), (ii + 1) * chunk_size)] for ii in
                      range(0, self._num_procs)]
            # TODO Überarbeiten Chunk aufteilung
            for i, chunk in enumerate(chunks):
                if not chunk:
                    # no empty chucks allowed
                    continue
                inqueue = Queue()
                outqueue = Queue()

                batch_list.append(AgentBatch(chunk,
                                             self.planning_problem_set, self.scenario,
                                             self.global_timestep,
                                             config_planner,
                                             self.config, self.msg_logger, self.log_path, self.mod_path,
                                             outqueue, inqueue))
        return batch_list

    def _set_collision_check(self):
        """
        Creates collision objects for the environment.
         self._cc_stat stores all static obstacles and the road boundary
         self._cc_dyn stores all collision objects for non-agent dynamic obstacles
        """

        self._cc_stat = []
        self._cc_dyn = []
        # create road boundary obstacle
        _, road_boundary_sg_obb = create_road_boundary_obstacle(self.scenario)

        for co in self.scenario.static_obstacles:
            # add static collision objects
            road_boundary_sg_obb.add_shape(create_collision_object(co))
        self._cc_stat = road_boundary_sg_obb

        for co in self.scenario.dynamic_obstacles:
            # add dynamic collision obstacles
            if co.obstacle_id in self.agent_id_list:
                continue
            self._cc_dyn.append(create_collision_object(co))

    def run_simulation(self):
        """ Starts the simulation.

        Wrapper function around parallel and sequential simulation
        to allow treating parallel and sequential simulations equally.
        """

        sim_time_start = time.perf_counter()
        # start agent batches
        if len(self.batch_list) == 1:
            # If we have only one batch, run sequential simulation
            self.run_sequential_simulation()

        else:
            # run parallel simulation
            self.run_parallel_simulation()

        self.process_times["complete_simulation"] = time.perf_counter() - sim_time_start
        self.msg_logger.info(f"Simulation completed")
        # TODO
        self.postprocess_simulation()
        self.process_times["total_elapsed_time"] = time.perf_counter() - sim_time_start

    def run_sequential_simulation(self):
        """
        As long as not all agents are finished, do a step in the sequential simulation for each non-finished agent
        and visualize current timestep if specified in the simulation configuration
        """
        running = True
        while running:
            self.process_times["simulation_steps"][self.global_timestep] = {}
            step_time_start = time.perf_counter()
            running = self.step_sequential_simulation()
            self.visualize_simulation(self.global_timestep)
            self.process_times["simulation_steps"][self.global_timestep].update({"total_sim_step" :time.perf_counter() - step_time_start})
            self.global_timestep += 1
        # get batch process times
        self.process_times[self.batch_list[0].name] = self.batch_list[0].process_times

    def run_parallel_simulation(self):
        """Control a simulation running in multiple processes.

        Starts processes, calls step_parallel_simulation,
        manages graceful termination and creates an animation from saved global plots.
        """
        # start parallel batches
        for batch in self.batch_list:
            batch.start()

        running = True
        while running:
            self.process_times["simulation_steps"][self.global_timestep] = {}
            step_time_start = time.perf_counter()
            running = self._step_parallel_simulation()
            self.process_times["simulation_steps"][self.global_timestep].update({"total_sim_step" :time.perf_counter() - step_time_start})
            self.global_timestep += 1

        # Workers should already have terminated, otherwise wait for timeouts
        self.msg_logger.info("Terminating workers...")

        for batch in self.batch_list:
            batch.in_queue.put("END", block=True)
            batch.join()

    def step_sequential_simulation(self):
        """
        Does a single simulation step in the single-process setting
        """

        # update Scenario, calculate new predictions and check for collision of prev. iteration
        self.global_predictions, colliding_agents = self.prestep_simulation()

        # perform single simulation step in agent batch
        self.batch_list[0].step_simulation(self.scenario, self.global_timestep, self.global_predictions,
                                           colliding_agents)

        running = not self.batch_list[0].finished
        return running

    def _step_parallel_simulation(self):
        """ Main function for stepping a parallel simulation.

        Computes the predictions, handles the communication with the agent batches,
        manages synchronization and termination of batches, and handles simulation-level
        logging and plotting.

        See also AgentBatch.run().

        :returns: running: True while the simulation has not completed.
        """

        # update Scenario, calculate new predictions and check for collision of prev. iteration
        self.global_predictions, colliding_agents = self.prestep_simulation()
        # send data to each batch process
        for batch in self.batch_list:
            batch.in_queue.put([self.scenario, self.global_timestep, self.global_predictions,
                                colliding_agents])

        # Plot previous timestep while batches are busy
        self.visualize_simulation(self.global_timestep - 1)

        # Receive simulation step results
        for batch in reversed(self.batch_list):
            try:
                # update agents
                queue_dict = batch.out_queue.get(block=True, timeout=TIMEOUT)
                agent_ids = list(queue_dict.keys())
                for agent_id in agent_ids:
                    agent = self.get_agent_by_id(agent_id)
                    for attr, value in queue_dict[agent_id].items():
                        if hasattr(agent, attr):
                            setattr(agent, attr, value)
                        else:
                            raise AttributeError
                # check if batch is finished
                [batch_finished, proc_times] = batch.out_queue.get(block=True, timeout=TIMEOUT)
                if batch_finished:
                    self.process_times[batch.name] = proc_times
                    # terminate finished process
                    batch.in_queue.put("END", block=True)
                    batch.join()
                    self.batch_list.remove(batch)
                    self.msg_logger.info(f"Closing Batch {batch.name}")

            except Empty:
                self.msg_logger.info("Timeout while waiting for step results!")
                return

        # Write global log
        # TODO
        # multi_agent_log.append_log(self.log_path, self.current_timestep, self.current_timestep * self.scenario.dt,
        #            step_time_end - step_time_start, sync_time_end - sync_time_start,
        #            self.agent_id_list, [self.agent_state_dict[id] for id in self.agent_id_list])

        return len(self.batch_list) > 0

    def prestep_simulation(self):
        preproc_time = time.perf_counter()
        self.msg_logger.info(f"Simulating timestep {self.global_timestep}")
        # check for collisions
        colliding_agents = self.check_collision()
        # update scenario
        self.update_scenario(colliding_agents)
        # Calculate new predictions
        predictions = ph.get_predictions(self.config, self._predictor, self.scenario, self.global_timestep,
                                         self.prediction_horizon)

        self.process_times["simulation_steps"][self.global_timestep].update({"preprocessing": time.perf_counter() - preproc_time})
        return predictions, colliding_agents

    def check_collision(self):
        coll_objects = []
        agent_ids = []
        collided_agents = []
        # get collision objects from agents
        for agent in self.agents:
            if agent.status == AgentStatus.RUNNING:
                coll_objects.append(agent.collision_objects[-1])
                agent_ids.append(agent.id)

        # check if any agent collides with a static obstacle / road boundary
        coll_time_stat = trajectory_queries.trajectories_collision_static_obstacles(coll_objects, self._cc_stat)
        if any(i > -1 for i in coll_time_stat):
            index = [i for i, n in enumerate(coll_time_stat) if n != -1]
            collided_agents.extend([agent_ids[i] for i in index])
            # TODO: check crash statistics in planner-> sollte funktionieren
        # # check if any agent collides with a dynamic obstacle
        coll_time_dyn = trajectory_queries.trajectories_collision_dynamic_obstacles(coll_objects, self._cc_dyn)
        if any(i > -1 for i in coll_time_dyn):
            index = [i for i, n in enumerate(coll_time_dyn) if n != -1]
            collided_agents.extend([agent_ids[i] for i in index])
            # TODO: check crash statistics s. planner -> sollte funktionieren

        # check if agents crash against each other
        if len(coll_objects) > 1:
            for index, ego in enumerate(coll_objects):
                other_agents = copy.copy(coll_objects)
                other_agents.remove(ego)
                coll_time = trajectory_queries.trajectories_collision_dynamic_obstacles([ego], other_agents,
                                                                                        method='box2d')
                if coll_time != [-1]:
                    # collision detected
                    collided_agents.append(agent_ids[index])
                    raise NotImplementedError("check functionality")  # TODO check functionality
                # TODO: check crash statistics s. planner -> sollte funktionieren
        return collided_agents

    def update_scenario(self, colliding_agents):
        if self.global_timestep == 0:
            # no need to update initial setting
            return
        shape = Rectangle(self.config.vehicle.length, self.config.vehicle.width)
        agents_to_update = [agent for agent in self.agents if agent.status > AgentStatus.IDLE]
        for agent in agents_to_update:
            if agent.status == AgentStatus.RUNNING and agent.id not in colliding_agents:
                obs = self.scenario.obstacle_by_id(agent.id)
                # add calculated position to trajectory of dummy obstacles
                if len(obs.prediction.trajectory.state_list) > 1:
                    initial_timestep = obs.prediction.trajectory.initial_time_step
                    traj = (obs.prediction.trajectory.states_in_time_interval(initial_timestep, self.global_timestep-1)
                            + agent.vehicle_history[-1].prediction.trajectory.state_list[1:])

                else:
                    traj = agent.vehicle_history[-1].prediction.trajectory.state_list[1:]

                lanelet_assigned = obs.prediction.center_lanelet_assignment
                obs.prediction = TrajectoryPrediction(
                    Trajectory(initial_time_step=traj[0].time_step, state_list=traj), shape,
                    center_lanelet_assignment=lanelet_assigned)
            elif agent.status == AgentStatus.COLLISION or agent.id in colliding_agents:
                raise NotImplementedError("Prediction still in trajectory?")
            elif agent.status == AgentStatus.COMPLETED:
                pass
            # TODO Cutoff trajectory if agent is finished/crashed? Currently ends with prediction (+2 seconds)
        return

    def postprocess_simulation(self):
        save = self.config_visu.save_all_final_trajectories
        show = self.config_visu.show_all_final_trajectories
        if save or show:
            visu.plot_multiagent_scenario_final_trajectories(self.scenario, self.agents,
                                                             self.config,
                                                             self.log_path,
                                                             show=show, save=save)
        if self.config_visu.save_gif:
            visu.make_gif(self.scenario, range(0, self.global_timestep - 1),
                          self.log_path, duration=0.1)

    # TODO
    # if not goal_status == "Scenario Successful!":
    #     with open(os.path.join(self.log_path, "log_failures.csv"), 'a') as file:
    #         line = str(self.scenario.scenario_id) + "\n"
    #         file.write(line)

    def visualize_simulation(self, timestep):
        if ((self.config_visu.show_plots or self.config_visu.save_plots or self.config_visu.save_gif)
                and self.config_simulation.use_multiagent):
            time_visu = time.perf_counter()
            visualize_multiagent_scenario_at_timestep(self.scenario,
                                                      self.agents,
                                                      timestep, self.config, self.log_path,
                                                      predictions=self.global_predictions,
                                                      plot_window=self.config_visu.plot_window_dyn,
                                                      save=self.config_visu.save_plots,
                                                      show=self.config_visu.show_plots,
                                                      gif=self.config_visu.save_gif)
            self.process_times["simulation_steps"][self.global_timestep].update({"time_visu": time.perf_counter()-time_visu})

    def get_agent_by_id(self, agent_id):
        """ Returns agent object by given id
        """
        [agent] = [i for i in self.agents if i.id == agent_id]
        return agent
