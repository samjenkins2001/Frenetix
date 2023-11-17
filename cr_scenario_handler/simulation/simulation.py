__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import time
import math
from multiprocessing import Queue
import psutil
from queue import Empty
from typing import List
import random
import copy

# commonroad-io
from commonroad.scenario.state import CustomState
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle
import commonroad_dc.pycrcc as pycrcc

# cr-scenario-handler
import cr_scenario_handler.utils.general as general
import cr_scenario_handler.utils.prediction_helpers as ph
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from cr_scenario_handler.utils.visualization import make_gif

from cr_scenario_handler.utils.configuration import Configuration
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import ObstacleType, ObstacleRole, DynamicObstacle
from commonroad.geometry.shape import Rectangle, Circle
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.planning.goal import GoalRegion
from commonroad.common.util import AngleInterval
from commonroad.common.util import Interval

from cr_scenario_handler.simulation.agent_batch import AgentBatch
from cr_scenario_handler.simulation.agent import Agent
import cr_scenario_handler.utils.multiagent_helpers as hf
from cr_scenario_handler.utils.multiagent_helpers import TIMEOUT, AgentStatus
import cr_scenario_handler.utils.multiagent_logging as multi_agent_log
from cr_scenario_handler.utils.visualization import visualize_multiagent_scenario_at_timestep, visualize_agent_at_timestep
from commonroad_dc.collision.trajectory_queries import trajectory_queries
# msg_logger = logging.getLogger("Simulation_logger")


class Simulation:

    def __init__(self, config_sim, config_planner):
        """ Main class for running a planner on a scenario.

        Manages the global configuration, creates and manages agent batches,
        handles global communication and coordination during parallel simulations,
        and does simulation-level logging and plotting.

        :param config: The configuration.
        :param log_path: Base path to write global log files to.
            Plots are written to <log_path>/plots/, and
            agent-level logs / plots are written to <log_path>/<agent_id>/ and
            <log_path>/<agent_id>/plots/.
        :param mod_path: Working directory of the planner, containing planner configuration.
        """

        # Configuration
        self.config = config_sim
        self.config_simulation = config_sim.simulation
        self.config_visu = config_sim.visualization
        # self.multiprocessing = config.multiagent.multiprocessing
        self.mod_path = self.config_simulation.mod_path
        self.log_path = self.config_simulation.log_path
        self._multiproc = self.config_simulation.multiprocessing
        # use specified number of processors, else use all physical cores
        self._num_procs = self.config_simulation.num_procs if not self.config_simulation.num_procs == -1 else (
            psutil.cpu_count(logical=False))

        self.global_timestep = 0

        self.msg_logger = multi_agent_log.logger_initialization(config_sim,  self.log_path, "Simulation_logger")
        self.msg_logger.critical("Start Scenario: " + self.log_path.split("/")[-1])

        self.scenario, self.agent_id_list, self.planning_problem_set = self._scenario_preprocessing()
        if self.config_simulation.use_multiagent:
            # add additional agents to agent_id_list and planning_problem_set
            multi_agent_id_list, multi_agent_planning_problems = self._multiagent_preprocessing()
            self.agent_id_list.extend(multi_agent_id_list)
            [self.planning_problem_set.add_planning_problem(agent_problem) for agent_problem in multi_agent_planning_problems]

        self.batch_list = self._create_agent_batches(config_planner)
        """List of tuples containing all batches and their associated fields:
        batch_list[i][0]: Agent Batch object
        batch_list[i][1]: Queue for sending data to the batch
        batch_list[i][2]: Queue for receiving data from the batch
        batch_list[i][3]: Agent IDs managed by the batch
        """

        self.running_agents_obs = []
        # Return values of the last agent step
        # self.dummy_obs = None
        self.agent_status_dict = dict()
        # [self.agent_state_dict.update(batch[0].agent_state_dict) for batch in self.batch_list]
        [self.agent_status_dict.update(batch.agent_status_dict) for batch in self.batch_list]

        self.agent_history_dict = dict.fromkeys(self.agent_id_list)
        self.agent_recorded_state_dict = dict.fromkeys(self.agent_id_list)
        self.agent_input_state_dict = dict.fromkeys(self.agent_id_list)
        self.agent_ref_path_dict = dict.fromkeys(self.agent_id_list)
        self.agent_traj_set_dict = dict.fromkeys(self.agent_id_list)

        self.global_predictions = None
        # get prediction horizon if specified in planner-configuration, else use 2 seconds by default
        self.prediction_horizon = config_planner.planning.planning_horizon if hasattr(config_planner, "planning") else 2

        # Additional Modules
        self._predictor = None
        self._occlusion = None
        self._reach_set = None

        self._load_external_modules()

        self._cc_dyn = None
        self._cc_stat = None
        self._set_collision_checker()

    def _scenario_preprocessing(self):
        """ Modify a commonroad scenario to prepare it for the simulation.

        Reads the scenario and planning problem from the configuration,
        selects the agents to be simulated, and creates missing planning problems and
        dummy obstacles.
        """
        planning_problems = []  # _set = PlanningProblemSet()
        agent_id_list = []
        scenario, _, original_planning_problem_set = general.load_scenario_and_planning_problem(self.config)

        # Add obstacles for original agents
        for problem in original_planning_problem_set.planning_problem_dict.values():
            dummy_obstacle, planning_problem = self._create_obstacle_for_planning_problem(problem)
            agent_id_list.append(planning_problem.planning_problem_id)
            # agent_obstacle_list.append(dummy_obstacle)
            scenario.add_objects(dummy_obstacle)
            planning_problems.append(planning_problem)

        return scenario, agent_id_list, PlanningProblemSet(planning_problems)

    def _multiagent_preprocessing(self):
        """ Modify a commonroad scenario to prepare it for the simulation.

        Reads the scenario and planning problem from the configuration,
        selects the agents to be simulated, and creates missing planning problems and
        dummy obstacles.
        """
        multi_agent_id_list = self._select_additional_agents()#scenario, agent_id_list)

        # Create PlanningProblems for additional agents
        multi_agent_planning_problems = self._create_planning_problems_for_agent_obstacles(multi_agent_id_list)

        # TODO WARUM?
        # for obs_id in multi_agent_id_list:
        #     obs = self.scenario.obstacle_by_id(obs_id)
        #     if obs.initial_state.time_step > 0:
        #         self.scenario.remove_obstacle(obs)

        return multi_agent_id_list, multi_agent_planning_problems

    def _select_additional_agents(self):
        """ Selects the dynamic obstacles that should be simulated as agents
        according to the multiagent configuration.

        :return: A List of obstacle IDs that should be used as agents
        """

        # Don't create additional agents if we run single agent
        # if not self.config.multiagent.use_multiagent:
        #     return []
        # Find all dynamic obstacles in the scenario
        allowed_types = [ObstacleType.CAR,
                         ObstacleType.TRUCK,
                         ObstacleType.BUS]
        allowed_roles = [ObstacleRole.DYNAMIC]
        allowed_id_list = [obs.obstacle_id for obs in self.scenario.obstacles if obs.obstacle_type in allowed_types and
                           obs.obstacle_role in allowed_roles and obs.obstacle_id not in self.agent_id_list]

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

    def _create_planning_problems_for_agent_obstacles(self, multi_agent_id_list):
        """ Creates the missing planning problems for agents that should be
        created from dynamic obstacles.

        The goal state is defined as a small area around the final state of the
        trajectory of the dynamic obstacle.
        The allowed deviations from this state are:
            time: +/- 5 time steps from final time step
            position: Circle with 3m diameter around final state
            velocity: +/- 2 m/s from final state
            orientation: +/- 20° from final state

        :return: obstacle_list, planning_problem_set
            Where obstacle_list is a list of the obstacles for which planning problems
                were created,
            and planning_problem_set is a new PlanningProblemSet with all created problems.
        """

        planning_problem_list = []
        for id in multi_agent_id_list:
            if not id in self.planning_problem_set.planning_problem_dict.keys():
                obstacle = self.scenario.obstacle_by_id(id)
                # agent_obstacle_list.append(obstacle)
                initial_state = obstacle.initial_state
                if not hasattr(initial_state, 'acceleration'):
                    initial_state.acceleration = 0.

                # create planning problem
                final_state = obstacle.prediction.trajectory.final_state
                goal_state = CustomState(time_step=Interval(final_state.time_step - 50, final_state.time_step + 50),
                                         position=Circle(1.5, final_state.position),
                                         velocity=Interval(final_state.velocity - 2, final_state.velocity + 2),
                                         orientation=AngleInterval(final_state.orientation - 0.349,
                                                                   final_state.orientation + 0.349))

                problem = PlanningProblem(id, initial_state, GoalRegion(list([goal_state])))
                planning_problem_list.append(problem)

        return planning_problem_list

    def _create_obstacle_for_planning_problem(self, planning_problem: PlanningProblem):
        """ Creates a dummy obstacle from a given planning problem.

        Extends the initial state of the planning problem to include all necessary values,
        and creates a DynamicObstacle from the initial state of the planning problem
        and the vehicle configuration.
        The prediction of the new obstacle contains only its current state.

        :param planning_problem: PlanningProblem to create a dummy obstacle for

        :return: dummy_obstacle, planning_problem
            Where dummy_obstacle is the generated DynamicObstacle,
            and planning_problem is the extended PlanningProblem.
        """

        id = planning_problem.planning_problem_id
        if not hasattr(planning_problem.initial_state, 'acceleration'):
            planning_problem.initial_state.acceleration = 0.

        # create dummy obstacle from the planning problem
        vehicle_params = self.config.vehicle
        shape = Rectangle(length=vehicle_params.length, width=vehicle_params.width)
        dummy_obstacle = DynamicObstacle(
            id,
            ObstacleType.CAR,
            shape,
            planning_problem.initial_state,
            TrajectoryPrediction(Trajectory(planning_problem.initial_state.time_step,
                                            [planning_problem.initial_state]),
                                 shape)
        )
        # otherwise prediction of this obstacle not available not used
        dummy_obstacle.prediction.final_time_step = planning_problem.goal.state_list[0].time_step.start

        return dummy_obstacle, planning_problem

    def _create_agent_batches(self, config_planner):
        """ Initialize the agent batches and set up the communication queues.

        Reads the configuration to determine the number of agent batches to create,
        creates the batches, and establishes the communication queues.

        :return: batch_list: List of agent batches and associated data, with
            batch_list[i][0]: AgentBatch object
            batch_list[i][1]: Queue for sending data to the batch
            batch_list[i][2]: Queue for receiving data from the batch
            batch_list[i][3]: Agent IDs managed by the batch
        """

        batch_list: List[AgentBatch | Agent] = []
        if not self._multiproc or self._num_procs < 3\
                or len(self.agent_id_list) < 2:
            # Multiprocessing disabled or useless, run single process
            batch_list.append(AgentBatch(self.agent_id_list, self.planning_problem_set, self.scenario,
                                          self.global_timestep, config_planner,
                                          self.config, self.msg_logger, self.log_path, self.mod_path))

        else:

            chunk_size = math.ceil(len(self.agent_id_list) / self._num_procs)
            chunks = [self.agent_id_list[ii * chunk_size:
                                         min(len(self.agent_id_list), (ii + 1) * chunk_size)] for ii in
                                         range(0, self._num_procs)]

            for i, chunk in enumerate(chunks):
                inqueue = Queue()
                outqueue = Queue()

                batch_list.append(AgentBatch(chunk, self.planning_problem_set, self.scenario, self.global_timestep,
                                             config_planner, self.config, self.msg_logger, self.log_path, self.mod_path,
                                             outqueue, inqueue))

        return batch_list

    def _load_external_modules(self):
        # Load prediction framework and reach set
        self._predictor = ph.load_prediction(self.scenario, self.config.prediction.mode)#, self.config)

        """---------------------------------------"""
        # TODO: include CR-Reach und CR-CriMe, Occlusion-Module,Sensormodel

        # *****************************
        # Load Reach Set
        # *****************************
        # if 'responsibility' in config.cost.cost_weights and config.cost.cost_weights['responsibility'] > 0:
        #     reach_set = ph.load_reachset(self.scenario, self.config, mod_path)

        """----------------------------------------"""

    def run_simulation(self):
        """ Starts the simulation.

        Wrapper function around run_parallel_simulation and AgentBatch.run_sequential
        to allow treating parallel and sequential simulations equally.
        """
        # multi_agent_log.init_log(self.log_path)
        sim_time_start = time.time()
        # start agent batches
        if len(self.batch_list) == 1:
            self.run_sequential_simulation()
            # If we have only one batch, run sequential simulation
            # self.batch_list[0][0].run_sequential(self.log_path, self.predictor, self.scenario)
        elif type(self.batch_list[0]) == AgentBatch:
            # start parallel batches
            for batch in self.batch_list:
                batch.start()

            # run parallel simulation
            self.run_parallel_simulation()

            # Workers should already have terminated, otherwise wait for timeouts
            self.msg_logger.info("[Simulation] Terminating workers...")
            for batch in self.batch_list:
                batch.join()

        else:
            raise TypeError("Simulation runs only with agents or agent batches!")

        self.msg_logger.info(f"Simulation completed")

        self.postprocess_simulation()

    def run_sequential_simulation(self):
        running = True
        while running:

            self.global_timestep = self.batch_list[0].global_timestep
            running = self.step_sequential_simulation()
            self._simulation_visualization_agent(self.global_timestep)

    def run_parallel_simulation(self):
        """Control a simulation running in multiple processes.

        Initializes the global log file, calls step_parallel_simulation,
        manages graceful termination and creates an animation from saved global plots.
        """
        running = True
        while running:

            running = self.step_parallel_simulation()
            self.global_timestep += 1

    def step_sequential_simulation(self):
        step_time_start = time.time()
        self.global_predictions = self.prestep_simulation()

        self.batch_list[0].step_simulation(self.global_predictions)

        self.agent_status_dict.update(self.batch_list[0].agent_status_dict)
        if any(i ==AgentStatus.COMPLETED for i in self.agent_status_dict.values()):
            bla = 0
            #TODO check why predictions are there one more timestep (predictions calculated based on prev. timestep?)
        self.agent_history_dict.update(self.batch_list[0].agent_history_dict)
        self.agent_recorded_state_dict.update(self.batch_list[0].agent_recorded_state_dict)
        self.agent_input_state_dict.update(self.batch_list[0].agent_input_state_dict)
        self.agent_ref_path_dict.update(self.batch_list[0].agent_ref_path_dict)

        self.update_simulation()

        colliding_agents = self.check_collision()

        self.batch_list[0].update_agents(self.scenario, colliding_agents)

        running = any([i < AgentStatus.COMPLETED for i in self.agent_status_dict.values()])
        return running

    def step_parallel_simulation(self):
        """ Main function for stepping a parallel simulation.

        Computes the predictions, handles the communication with the agent batches,
        manages synchronization and termination of batches, and handles simulation-level
        logging and plotting.

        See also AgentBatch.run().

        :returns: running: True while the simulation has not completed.
        """

        # START TIMER
        step_time_start = time.time()

        self.global_predictions = self.prestep_simulation()
        # # Calculate new predictions

        # Send predictions to agent batches
        for batch in self.batch_list:
            batch.in_queue.put(self.global_predictions)

        # Plot previous timestep while batches are busy
        # Remove agents that did not exist in the last timestep
        self._simulation_visualization_agent(self.global_timestep-1)

        # Receive simulation step results
        for batch in reversed(self.batch_list):
            try:
                agent_status_batch = batch.out_queue.get(block=True, timeout=TIMEOUT)
                self.agent_status_dict.update(agent_status_batch)
                self.agent_history_dict.update(batch.out_queue.get(block=True, timeout=TIMEOUT))
                self.agent_recorded_state_dict.update(batch.out_queue.get(block=True, timeout=TIMEOUT))
                self.agent_input_state_dict.update(batch.out_queue.get(block=True, timeout=TIMEOUT))
                self.agent_ref_path_dict.update(batch.out_queue.get(block=True, timeout=TIMEOUT))
                # self.agent_traj_set_dict.update(batch.out_queue.get(block=True, timeout=TIMEOUT))

                if all([i > AgentStatus.RUNNING for i in agent_status_batch.values()]):
                    batch.in_queue.put("END", block=True)
                    batch.join()
                    self.batch_list.remove(batch)
            except Empty:
                self.msg_logger.info(" Timeout while waiting for step results!")
                return

        self.update_simulation()

        colliding_agents = self.check_collision()
        for batch in self.batch_list:
            batch.in_queue.put([self.scenario, colliding_agents])

        terminated_batch_list = []

        # STOP TIMER
        step_time_end = time.time()

        return len(self.batch_list) > 0

    def prestep_simulation(self):
        self.msg_logger.info(f"Simulating timestep {self.global_timestep}")
        # Calculate new predictions
        predictions = ph.get_predictions(self.config, self._predictor, self.scenario, self.global_timestep, self.prediction_horizon)
        return predictions

    def postprocess_simulation(self):
        if self.config.debug.gif:
            make_gif(self.config, self.scenario, range(0, self.global_timestep - 1),
                     self.log_path, duration=0.1)

    def _set_collision_checker(self):#, collision_checker: pycrcc.CollisionChecker = None):
        """
        Sets the collision checker used by the planner using either of the two options:
        If a collision_checker object is passed, then it is used directly by the planner.
        If no collision checker object is passed, then a CommonRoad scenario must be provided from which the collision
        checker is created and set.
        :param scenario: CommonRoad Scenario object
        :param collision_checker: pycrcc.CollisionChecker object"""
        self._cc_stat = []
        self._cc_dyn = []
        _, road_boundary_sg_obb = create_road_boundary_obstacle(self.scenario)
        for co in self.scenario.static_obstacles:
            # self._cc_stat.append(create_collision_object(co))
            road_boundary_sg_obb.add_shape(create_collision_object(co))
        self._cc_stat = road_boundary_sg_obb
        # _, road_boundary_sg_obb = create_road_boundary_obstacle(self.scenario)
        # self._cc_stat.append(road_boundary_sg_obb)
        for co in self.scenario.dynamic_obstacles:
            if co.obstacle_id in self.agent_id_list:
                continue
            self._cc_dyn.append(create_collision_object(co))

    def check_collision(self):
        coll_objects = []
        agents = []
        collided_agents = []
        for agent, occupancy in self.agent_history_dict.items():
            if self.agent_status_dict[agent] == AgentStatus.RUNNING:
                ego = pycrcc.TimeVariantCollisionObject(self.global_timestep)
                ego.append_obstacle(pycrcc.RectOBB(0.5 * self.config.vehicle.length, 0.5 * self.config.vehicle.width,
                                                   occupancy[-1].initial_state.orientation,
                                                   occupancy[-1].initial_state.position[0],
                                                   occupancy[-1].initial_state.position[1]))
                coll_objects.append(ego)
                agents.append(agent)

                # i = create_collision_object(occupancy[-1])
                # coll_objects.append(create_collision_object(occupancy[-1], self.config.vehicle, self.agent_recorded_state_dict[agent][-1]))
        # check if any agent collides with a static obstacle / road boundary
        coll_time_stat = trajectory_queries.trajectories_collision_static_obstacles(coll_objects, self._cc_stat)
        if any(i > -1 for i in coll_time_stat):
            index = [i for i, n in enumerate(coll_time_stat) if n != -1]
            collided_agents.extend([agents[i] for i in index])
            # TODO: check crash statistics s. planner
        coll_time_dyn = trajectory_queries.trajectories_collision_dynamic_obstacles(coll_objects, self._cc_dyn)
        if any(i > -1 for i in coll_time_dyn):
            index = [i for i, n in enumerate(coll_time_dyn) if n != -1]
            collided_agents.extend([agents[i] for i in index])
            # TODO: check crash statistics s. planner

        # check if agents crash against each other
        if len(coll_objects)>1:
            for index, agent in enumerate(coll_objects):
                other_agents = copy.copy(coll_objects)
                other_agents.remove(agent)
                coll_time = trajectory_queries.trajectories_collision_dynamic_obstacles([agent], other_agents,
                                                                                        method='box2d')
                if coll_time != [-1]:
                    # collision detected

                    collided_agents.append(agents[index])
                    raise NotImplementedError  # TODO check functionality
                # TODO check crash statistics s. planner
        return collided_agents

    def update_simulation(self):

        agents_to_update = [agent_id for agent_id, status in self.agent_status_dict.items() if status>AgentStatus.INITIAL]
        self.running_agents_obs = [history[-1] for agent, history in self.agent_history_dict.items() if self.agent_status_dict[agent] == AgentStatus.RUNNING]
        self.scenario = hf.scenario_without_obstacle_id(self.scenario, agents_to_update)
        self.scenario.add_objects(self.running_agents_obs)
        # self._update_scenario()#dummy_obs_dict)#, agents_to_update)

    def _simulation_visualization_agent(self, timestep):
        if ((self.config_visu.show_plots or self.config_visu.save_plots or self.config_visu.save_gif) and
                len(self.running_agents_obs) > 0) and self.config.simulation.use_multiagent:
            visualize_multiagent_scenario_at_timestep(self.scenario, self.planning_problem_set,
                                                      self.running_agents_obs, timestep, self.config, self.log_path,
                                                      traj_set_dict=self.agent_traj_set_dict,
                                                      ref_path_dict=self.agent_ref_path_dict,
                                                      predictions=self.global_predictions,
                                                      plot_window=self.config_visu.plot_window_dyn)


