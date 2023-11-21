__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import traceback
from copy import deepcopy
from typing import List

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType

from commonroad_dc import pycrcc

from frenetix_motion_planner.reactive_planner_cpp import ReactivePlannerCpp
from frenetix_motion_planner.reactive_planner import ReactivePlannerPython
from frenetix_motion_planner.state import ReactivePlannerState
from frenetix_motion_planner.occlusion_planning.occlusion_module import OcclusionModule

from cr_scenario_handler.utils import helper_functions as hf
import cr_scenario_handler.utils.multiagent_logging as lh
from cr_scenario_handler.utils.utils_coordinate_system import extend_ref_path
from cr_scenario_handler.planner_interfaces.planner_interface import PlannerInterface
from cr_scenario_handler.utils.collision_report import coll_report

from behavior_planner.behavior_module import BehaviorModule
from commonroad_route_planner.route_planner import RoutePlanner


class FrenetPlannerInterface(PlannerInterface):

    def __init__(self, agent_id: int, config_planner, config_sim, scenario: Scenario,
                 planning_problem: PlanningProblem, log_path: str, mod_path: str):
        """ Class for using the frenetix_motion_planner Frenet planner with the cr_scenario_handler.

        Implements the PlannerInterface.

        :param config: The configuration object.
        :param scenario: The scenario to be solved. May not contain the ego obstacle.
        :param planning_problem: The planning problem of the ego vehicle.
        :param log_path: Path for writing planner-specific log files to.
        :param mod_path: Working directory of the planner.
        """
        self.config_plan = config_planner
        self.config_sim = config_sim
        self.scenario = scenario
        self.id = agent_id
        self.config_sim.simulation.ego_agent_id = agent_id
        self.DT = self.config_plan.planning.dt

        self.planning_problem = planning_problem
        self.log_path = log_path
        self.mod_path = mod_path

        # *************************************
        # Message Logger of Run
        # *************************************
        # self.msg_logger = logging.getLogger("Message_logger_" +str(self.id ))
        self.msg_logger = lh.logger_initialization(self.config_plan, log_path, "Message_logger_" + str(self.id))
        self.msg_logger.critical("Start Planner Vehicle ID: " + str(self.id))
        # Init and Goal State

        # Initialize planner
        self.planner = ReactivePlannerCpp(self.config_plan, self.config_sim, scenario, planning_problem, log_path, mod_path, self.msg_logger) \
            if self.config_plan.debug.use_cpp else \
              ReactivePlannerPython(self.config_plan, self.config_sim, scenario, planning_problem, log_path, mod_path, self.msg_logger)

        problem_init_state = planning_problem.initial_state

        if not hasattr(problem_init_state, 'acceleration'):
            problem_init_state.acceleration = 0.
        x_0 = deepcopy(problem_init_state)

        shape = Rectangle(self.planner.vehicle_params.length, self.planner.vehicle_params.width)
        ego_vehicle = DynamicObstacle(agent_id, ObstacleType.CAR, shape, x_0, None)
        self.planner.set_ego_vehicle_state(current_ego_vehicle=ego_vehicle)

        # Set initial state and curvilinear state
        self.x_0 = ReactivePlannerState.create_from_initial_state(
            deepcopy(planning_problem.initial_state),
            self.config_sim.vehicle.wheelbase,
            self.config_sim.vehicle.wb_rear_axle
        )
        self.planner.record_state_and_input(self.x_0)

        self.x_cl = None
        self.desired_velocity = None
        self.occlusion_module = None
        self.behavior_module = None
        self.route_planner = None

        # Set reference path
        if not self.config_sim.behavior.use_behavior_planner:
            route_planner = RoutePlanner(scenario=scenario, planning_problem=planning_problem)
            self.reference_path = route_planner.plan_routes().retrieve_first_route().reference_path
            self.reference_path, _ = route_planner.extend_reference_path_at_start(reference_path=self.reference_path,
                                                                                  initial_position_cart=self.x_0.position,
                                                                                  additional_lenght_in_meters=10.0)
        else:
            raise NotImplementedError

        # **************************
        # Initialize Occlusion Module
        # **************************
        # if config.occlusion.use_occlusion_module:
        #     self.occlusion_module = OcclusionModule(scenario, config, reference_path, log_path, self.planner)

        # **************************
        # Set External Planner Setups
        # **************************
        self.planner.update_externals(x_0=self.x_0, reference_path=self.reference_path)
        self.x_cl = self.planner.x_cl

    @property
    def all_traj(self):
        """Return the sampled trajectory bundle for plotting purposes."""
        return self.planner.all_traj

    @property
    def ref_path(self):
        """Return the reference path for plotting purposes."""
        return self.planner.reference_path

    def check_collision(self, ego_vehicle_list: List[DynamicObstacle], timestep: int):
        """ Check for collisions with the ego vehicle.

        Adapted from ReactivePlanner.check_collision to allow using ego obstacles
        that contain the complete (past and future) trajectory.

        :param ego_vehicle_list: List containing the ego obstacles from at least
            the last two time steps.
        :param timestep: Time step to check for collisions at.

        :return: True iff there was a collision.
        """
        raise EnvironmentError
    #     ego_vehicle = ego_vehicle_list[-1]
    #
    #     ego = pycrcc.TimeVariantCollisionObject((timestep + 1))
    #     ego.append_obstacle(
    #         pycrcc.RectOBB(0.5 * self.planner.vehicle_params.length, 0.5 * self.planner.vehicle_params.width,
    #                        ego_vehicle.state_at_time(timestep).orientation,
    #                        ego_vehicle.state_at_time(timestep).position[0],
    #                        ego_vehicle.state_at_time(timestep).position[1]))
    #
    #     if not self.planner.collision_checker.collide(ego):
    #         return False
    #     else:
    #         try:
    #             goal_position = []
    #
    #             if self.planner.goal_checker.goal.state_list[0].has_value("position"):
    #                 for x in self.planner.reference_path:
    #                     if self.planner.goal_checker.goal.state_list[0].position.contains_point(x):
    #                         goal_position.append(x)
    #                 s_goal_1, d_goal_1 = self.planner._co.convert_to_curvilinear_coords(
    #                     goal_position[0][0],
    #                     goal_position[0][1])
    #                 s_goal_2, d_goal_2 = self.planner._co.convert_to_curvilinear_coords(
    #                     goal_position[-1][0],
    #                     goal_position[-1][1])
    #                 s_goal = min(s_goal_1, s_goal_2)
    #                 s_start, d_start = self.planner._co.convert_to_curvilinear_coords(
    #                     self.planner.planning_problem.initial_state.position[0],
    #                     self.planner.planning_problem.initial_state.position[1])
    #                 s_current, d_current = self.planner._co.convert_to_curvilinear_coords(
    #                     ego_vehicle.state_at_time(timestep).position[0],
    #                     ego_vehicle.state_at_time(timestep).position[1])
    #                 progress = ((s_current - s_start) / (s_goal - s_start))
    #             elif "time_step" in self.planner.goal_checker.goal.state_list[0].attributes:
    #                 progress = (timestep - 1 / self.planner.goal_checker.goal.state_list[0].time_step.end)
    #             else:
    #                 print('Could not calculate progress')
    #                 progress = None
    #         except:
    #             progress = None
    #             print('Could not calculate progress')
    #             traceback.print_exc()
    #
    #         collision_obj = self.planner.collision_checker.find_all_colliding_objects(ego)[0]
    #         if isinstance(collision_obj, pycrcc.TimeVariantCollisionObject):
    #             obj = collision_obj.obstacle_at_time(timestep)
    #             center = obj.center()
    #             last_center = collision_obj.obstacle_at_time(timestep - 1).center()
    #             r_x = obj.r_x()
    #             r_y = obj.r_y()
    #             orientation = obj.orientation()
    #             self.planner.logger.log_collision(True, self.planner.vehicle_params.length,
    #                                               self.planner.vehicle_params.width,
    #                                               progress, center,
    #                                               last_center, r_x, r_y, orientation)
    #         else:
    #             self.planner.logger.log_collision(False, self.planner.vehicle_params.length,
    #                                               self.planner.vehicle_params.width,
    #                                               progress)
    #
    #         if self.config.debug.collision_report:
    #             coll_report(ego_vehicle_list, self.planner, self.scenario,
    #                         self.planning_problem, timestep, self.config, self.log_path)
    #
    #         return True

    def update_planner(self, scenario: Scenario, predictions: dict):
        """ Update the planner before the next time step.

        Updates the scenario and the internal states, and sets the new predictions.

        :param scenario: Updated scenario reflecting the new positions of other agents.
        :param predictions: Predictions for the other obstacles in the next time steps.
        """
        self.scenario = scenario

        # TODO Behavior Planner in simulation oder hier?
        if not self.config_sim.behavior.use_behavior_planner:
            # set desired velocity
            self.desired_velocity = hf.calculate_desired_velocity(scenario, self.planning_problem, self.x_0, self.DT,
                                                             desired_velocity=self.desired_velocity)
        else:
            raise NotImplementedError
            # behavior = behavior_modul.execute(predictions=predictions, ego_state=x_0, time_step=current_count)
            # desired_velocity = behavior_modul.desired_velocity
            # self.reference_path = behavior_modul.reference_path
        # End TODO

        self.planner.update_externals(scenario=scenario, x_0=self.x_0, x_cl=self.x_cl,
                                      desired_velocity=self.desired_velocity, predictions=predictions, occlusion_module=None)

    def plan(self, current_timestep):
        """ Execute one planing step.

        update_planner has to be called before this function.
        Plans the trajectory for the next time step, updates the
        internal state of the FrenetInterface, and shifts the trajectory
        to the global representation.

        :return: error, trajectory
            where error is:
                0: If an optimal trajectory has been found.
                1: Otherwise.
            and trajectory is:
                A Trajectory object containing the planned trajectory,
                    using the vehicle center for the position: If error == 0
                None: Otherwise
        """
        # plan trajectory
        optimal_trajectory_pair = self.planner.plan()

        if not optimal_trajectory_pair:
            # Could not plan feasible trajectory
            self.msg_logger.critical("No Kinematic Feasible and Optimal Trajectory Available!")
            return None

        # record the new state for planner-internal logging
        self.planner.record_state_and_input(optimal_trajectory_pair[0].state_list[1])

        # update init state and curvilinear state
        self.x_0 = deepcopy(self.planner.record_state_list[-1])
        self.x_cl = (optimal_trajectory_pair[2][1], optimal_trajectory_pair[3][1])

        self.msg_logger.info(f"current time step: {current_timestep}")
        self.msg_logger.info(f"current velocity: {self.x_0.velocity}")
        self.msg_logger.info(f"current target velocity: {self.desired_velocity}")

        return optimal_trajectory_pair[0]

