__author__ = "Maximilian Streubel, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# standard imports
import time
from copy import deepcopy
from typing import List
import inspect
import logging
import os

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InputState
from commonroad.scenario.trajectory import Trajectory
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction

# cr_scenario_handler
import cr_scenario_handler.utils.prediction_helpers as ph
import cr_scenario_handler.utils.multiagent_helpers as hf
import cr_scenario_handler.utils.goalcheck as gc
from cr_scenario_handler.utils.visualization import visualize_agent_at_timestep, make_gif
from cr_scenario_handler.utils.multiagent_helpers import AgentStatus
import cr_scenario_handler.planner_interfaces as planner_interfaces
from cr_scenario_handler.planner_interfaces.planner_interface import PlannerInterface

from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle

# TODO aus frenetix raus iwie
from frenetix_motion_planner.state import ReactivePlannerState

# msg_logger = logging.getLogger("Simulation_logger")


class Agent:

    def __init__(self, agent_id: int, planning_problem: PlanningProblem,
                 scenario: Scenario, config_planner, config_sim, msg_logger):
        """Represents one agent of a multiagent or single-agent simulation.

        Manages the agent's local view on the scenario, the planning problem,
        planner interface, collision detection, and per-agent plotting and logging.
        Contains the step function of the agent.

        :param agent_id: The agent ID, equal to the obstacle_id of the
            DynamicObstacle it is represented by.
        :param planning_problem: The planning problem of this agent.
        :param config: The configuration.
        :param log_path: Path for logging and visualization.
        :param mod_path: working directory of the planner, containing planner configuration.
        """
        # Agent id, equals the id of the dummy obstacle
        self.config = config_sim
        self.config_simulation = config_sim.simulation
        self.config_visu = config_sim.visualization

        self.mod_path = config_sim.simulation.mod_path

        self.log_path = os.path.join(config_sim.simulation.log_path, str(agent_id)) if (
                                     self.config_simulation.use_multiagent) else config_sim.simulation.log_path

        self.msg_logger = msg_logger
        self.vehicle = config_sim.vehicle

        self.dt = scenario.dt
        self.id = agent_id
        self.status = AgentStatus.INITIAL
        self.crashed = False
        self.scenario = None
        self.update_agent(scenario=deepcopy(scenario))

        self.record_state_list = list()
        # List of input states for the planner
        self.record_input_list = list()
        self.vehicle_history = list()

        self.planning_times = list()

        self.goal_checker = gc.GoalReachedChecker(planning_problem)
        self.goal_status = None
        self.goal_message = None
        self.full_goal_status = None

        self.predictions = None
        self.visible_area = None

        # Initialize Planning Problem
        self.planning_problem = planning_problem
        problem_init_state = deepcopy(planning_problem.initial_state)

        if not hasattr(problem_init_state, 'acceleration'):
            problem_init_state.acceleration = 0.
        x_0 = deepcopy(problem_init_state)

        shape = Rectangle(self.vehicle.length, self.vehicle.width)
        ego_vehicle = DynamicObstacle(planning_problem.planning_problem_id, ObstacleType.CAR, shape, x_0, None)
        self.set_ego_vehicle_state(ego_vehicle)

        x_0 = ReactivePlannerState.create_from_initial_state(problem_init_state, self.vehicle.wheelbase,
                                                             self.vehicle.wb_rear_axle)
        self.record_state_and_input(x_0)

        # Initialize Time Variables
        self.current_timestep = self.planning_problem.initial_state.time_step
        try:
            self.max_time_steps_scenario = int(self.config_simulation.max_steps*planning_problem.goal.state_list[0].time_step.end)
        except NameError:
            self.max_time_steps_scenario = 200

        # Initialize Planner
        used_planner = self.config_simulation.used_planner_interface
        try:
            planner_interface = [cls for _, module in inspect.getmembers(planner_interfaces, inspect.ismodule)
                  for name, cls in inspect.getmembers(module, inspect.isclass) if issubclass(cls, PlannerInterface)
                  and name == used_planner][0]
        except:
            raise ModuleNotFoundError(f"No such planner class found in planner_interfaces: {used_planner}")
        self.planner_interface = planner_interface(agent_id, config_planner, config_sim, scenario, planning_problem,
                                         self.log_path, self.mod_path)

        if config_sim.occlusion.use_occlusion_module:
            raise NotImplementedError
            # TODO add here instead of in interface
        return

    @property
    def reference_path(self):
        return self.planner_interface.ref_path

    @property
    def traj_set(self):
        return self.planner_interface.all_traj

    # def initialize_state_list(self):
    #     """ Initialize the recorded trajectory of the agent.
    #
    #     Fills the state list before the agent's initial time step with empty states,
    #     and creates and inserts the initial state of the agent.
    #     """
    #
    #     # In case of late startup, fill history with empty states
    #     for i in range(self.current_timestep):
    #         self.record_state_list.append(
    #             CustomState(time_step=i,
    #                         position=np.array([float("NaN"), float("NaN")]),
    #                         steering_angle=0, velocity=0, orientation=0,
    #                         acceleration=0, yaw_rate=0)
    #         )
    #     # Convert initial state to required format, append it to the state list
    #     self.record_state_list.append(
    #         CustomState(time_step=self.planning_problem.initial_state.time_step,
    #                     position=self.planning_problem.initial_state.position,
    #                     steering_angle=0,
    #                     velocity=self.planning_problem.initial_state.velocity,
    #                     orientation=self.planning_problem.initial_state.orientation,
    #                     acceleration=self.planning_problem.initial_state.acceleration,
    #                     yaw_rate=self.planning_problem.initial_state.yaw_rate)
    #     )

    def update_agent(self, scenario:Scenario, collision: bool=False):
        """ Update the scenario to synchronize the agents.

        :param outdated_agents: Obstacle IDs of all dummy obstacles that need to be updated
        :param dummy_obstacles: New dummy obstacles
        """
        self.scenario = hf.scenario_without_obstacle_id(scenario=deepcopy(scenario), obs_ids=[self.id])
        self.crashed = collision

    def check_goal_reached(self):
        """Check for completion of the planner.

        :return: True iff the goal area has been reached.
        """
        # TODO use planning_problem.goal_reached() ?

        # self.planning_problem.goal_reached() ->geht nur mit Trajektory
        self.goal_checker.register_current_state(self.record_state_list[-1])
        self.goal_status, self.goal_message, self.full_goal_status = self.goal_checker.goal_reached_status()

    def step_agent(self, global_predictions):
        """ Execute one planning step.

        Checks for collisions, filters the predictions by visibility,
        calls the step function of the planner, extends the recorded trajectory,
        creates the updated dummy obstacle for synchronization,
        records planning times and handles per-agent plotting.

        :param global_predictions: Dictionary of predictions for all obstacles and agents

        :returns: status, ego_obstacle
            where status is AgentStatus:
                0: if successful.
                1: if completed.
                2: on timelimit
                3: on error.
                4: on collision.
            and ego_obstacle is:
                The dummy obstacle at the new position of the agent,
                    including both history and planned trajectory,
                or None if status > 0
        """

        if self.crashed:
            self.msg_logger.info(f"Agent {self.id}: Collision Detected in timestep {self.current_timestep}!")
            self.status = AgentStatus.COLLISION
        elif self.current_timestep > self.max_time_steps_scenario:
            self.msg_logger.info(f"Agent {self.id}: Timelimit reached!")
            self.status = AgentStatus.TIMELIMIT
        else:
            self.check_goal_reached()
            if self.goal_status:
                self.msg_logger.info(f"Agent {self.id}: Scenario completed!")
                self.status = AgentStatus.COMPLETED

            # check for completion of this agent
            else:
                self.current_timestep = len(self.record_state_list) - 1
                self.msg_logger.info(f"Agent {self.id} current time step: {self.current_timestep}")

                # TODO: Occlusion module
                self.predictions, self.visible_area = ph.filter_global_predictions(self.scenario, global_predictions, self.vehicle_history[-1],
                                                                                   self.current_timestep, self.config,occlusion_module=None,
                                                                                   ego_id=self.id, msg_logger=self.msg_logger)

                # **************************
                # Cycle Occlusion Module
                # **************************
                # if config.occlusion.use_occlusion_module:
                #     occlusion_module.step(predictions=predictions, x_0=planner.x_0, x_cl=planner.x_cl)
                #     predictions = occlusion_module.predictions

                # **************************
                # Set Planner Subscriptions
                # **************************
                self.planner_interface.update_planner(self.scenario, self.predictions)

                # **************************
                # Execute Planner
                # **************************
                comp_time_start = time.time()
                trajectory = self.planner_interface.plan(self.current_timestep)
                comp_time_end = time.time()
                # END TIMER
                self.planning_times.append(comp_time_end - comp_time_start)
                self.msg_logger.info(f"Agent {self.id}: Total Planning Time: \t\t{self.planning_times[-1]:.5f} s")

                if trajectory:
                    # self.optimal
                    self.record_state_and_input(trajectory.state_list[1])
                    current_ego_vehicle = self.convert_state_list_to_commonroad_object(trajectory.state_list)
                    self.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)
                    # dummy[self.id] = current_ego_vehicle
                    self.status = AgentStatus.RUNNING

                    # plot own view on scenario
                    if self.id in self.config_visu.show_specific_individual_plots or self.config_visu.show_all_individual_plots \
                            or self.id in self.config_visu.save_specific_individual_plots or self.config_visu.save_all_individual_plots \
                            or ((self.config_visu.save_plots or self.config_visu.show_plots) and not self.config.simulation.use_multiagent):
                        visualize_agent_at_timestep(self.scenario, self.planning_problem,
                                                    self.vehicle_history[-1], self.current_timestep,
                                                    self.config, self.log_path,
                                                    traj_set=self.traj_set if self.traj_set else None,
                                                    optimal_traj=self.planner_interface.planner.trajectory_pair[0]
                                                    if self.planner_interface.planner.trajectory_pair else None,
                                                    ref_path=self.planner_interface.reference_path,
                                                    predictions=self.predictions,
                                                    visible_area=self.visible_area,
                                                    plot_window=self.config_visu.plot_window_dyn)

                else:
                    self.msg_logger.critical(f"Agent {self.id}: No Kinematic Feasible and Optimal Trajectory Available!")
                    self.status = AgentStatus.ERROR

    def record_state_and_input(self, state):
        """
        Adds state to list of recorded states
        Adds control inputs to list of recorded inputs
        """
        # append state to state list
        self.record_state_list.append(state)

        # compute control inputs and append to input list
        if len(self.record_state_list) > 1:
            steering_angle_speed = (state.steering_angle - self.record_state_list[-2].steering_angle) / self.dt
        else:
            steering_angle_speed = 0.0

        input_state = InputState(time_step=state.time_step,
                                 acceleration=state.acceleration,
                                 steering_angle_speed=steering_angle_speed)
        self.record_input_list.append(input_state)

    def set_ego_vehicle_state(self, current_ego_vehicle):
        self.vehicle_history.append(current_ego_vehicle)

    def convert_state_list_to_commonroad_object(self, state_list: List[ReactivePlannerState]):
        """
        Converts a CR trajectory to a CR dynamic obstacle with given dimensions
        :param state_list: trajectory state list of reactive planner
        :param obstacle_id: [optional] ID of ego vehicle dynamic obstacle
        :return: CR dynamic obstacle representing the ego vehicle
        """
        # shift trajectory positions to center
        new_state_list = list()
        for state in state_list:
            new_state_list.append(state.shift_positions_to_center(self.config.vehicle.wb_rear_axle))

        trajectory = Trajectory(initial_time_step=new_state_list[0].time_step, state_list=new_state_list)
        # get shape of vehicle
        shape = Rectangle(self.config.vehicle.length, self.config.vehicle.width)
        # get trajectory prediction
        prediction = TrajectoryPrediction(trajectory, shape)
        return DynamicObstacle(self.id, ObstacleType.CAR, shape, trajectory.state_list[0], prediction)
