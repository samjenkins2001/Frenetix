__author__ = "Rainer Trauth, Gerald Würsching, Christian Pek"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# python packages
import math
import time

import numpy as np
from typing import List, Optional, Tuple
import logging
import multiprocessing
from risk_assessment.risk_costs import calc_risk
from risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle
from multiprocessing.context import Process
from omegaconf import OmegaConf

# commonroad-io
from commonroad.common.validity import *
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InputState
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.planning_problem import GoalRegion

# commonroad_dc
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectory_preprocess_obb_sum, \
                                                                          trajectories_collision_static_obstacles

# commonroad_rp imports
from commonroad_rp.sampling_matrix import SamplingHandler
from commonroad_rp.polynomial_trajectory import QuinticTrajectory, QuarticTrajectory
from commonroad_rp.trajectories import TrajectoryBundle, TrajectorySample, CartesianSample, CurviLinearSample
from commonroad_rp.utility.utils_coordinate_system import CoordinateSystem, interpolate_angle

from commonroad_rp.state import ReactivePlannerState
from commonroad_rp.prediction_helpers import collision_checker_prediction

from commonroad_rp.cost_functions.cost_function import AdaptableCostFunction
from cr_scenario_handler.utils.goalcheck import GoalReachedChecker
from cr_scenario_handler.utils.configuration import Configuration
from commonroad_rp.utility.logging_helpers import DataLoggingCosts

from commonroad_rp.utility import helper_functions as hf

from commonroad_rp.utility.load_json import (
    load_harm_parameter_json,
    load_risk_json
)

# precision value
_EPS = 1e-5

# get logger
msg_logger = logging.getLogger("Message_logger")


class ReactivePlanner(object):
    """
    Reactive planner class that plans trajectories in a sampling-based fashion
    """
    def __init__(self, config: Configuration, scenario, planning_problem, log_path, work_dir):
        """
        Constructor of the reactive planner
        : param config: Configuration object holding all planner-relevant configurations
        """
        # Set horizon variables
        self.config = config
        self.horizon = config.planning.planning_horizon
        self.dT = config.planning.dt
        self.N = int(config.planning.planning_horizon / config.planning.dt)
        self._check_valid_settings()
        self.vehicle_params = config.vehicle
        self._low_vel_mode_threshold = config.planning.low_vel_mode_threshold

        # Multiprocessing & Settings
        self._multiproc = config.debug.multiproc
        self._num_workers = config.debug.num_workers

        # Initial State
        self.x_0: Optional[ReactivePlannerState] = None
        self.x_cl: Optional[Tuple[List, List]] = None

        self.record_state_list: List[ReactivePlannerState] = list()
        self.record_input_list: List[InputState] = list()

        self.ego_vehicle_history = list()
        self._LOW_VEL_MODE = False

        # Scenario
        self._co: Optional[CoordinateSystem] = None
        self._cc: Optional[pycrcc.CollisionChecker] = None
        self.scenario = None
        self.road_boundary = None
        self.set_scenario(scenario)
        self.planning_problem = planning_problem
        self.predictions = None
        self.reach_set = None
        self.behavior = None
        self.set_new_ref_path = None
        self.cost_function = None
        self.goal_status = False
        self.full_goal_status = None
        self.goal_area = hf.get_goal_area_shape_group(planning_problem=planning_problem, scenario=scenario)
        self.occlusion_module = None
        self.goal_message = "Planner is in time step 0!"
        self.use_amazing_visualizer = config.debug.use_amazing_visualizer

        self._desired_speed = None
        self._desired_d = 0.
        self.max_seen_costs = 1
        self.cost_weights = OmegaConf.to_object(config.cost.cost_weights)

        # **************************
        # Extensions Initialization
        # **************************
        if config.prediction.mode:
            self.use_prediction = True
        else:
            self.use_prediction = False

        self._goal_checker = GoalReachedChecker(planning_problem)

        # **************************
        # Statistics Initialization
        # **************************
        self._total_count = 0
        self._infeasible_count_collision = 0
        self._infeasible_count_kinematics = np.zeros(10)
        self.infeasible_kinematics_percentage = None
        self._optimal_cost = 0

        # **************************
        # Sampling Initialization
        # **************************
        # Set Sampling Parameters#
        self._sampling_min = config.sampling.sampling_min
        self._sampling_max = config.sampling.sampling_max
        self.sampling_handler = SamplingHandler(dt=self.dT, max_sampling_number=config.sampling.sampling_max,
                                                t_min=config.sampling.t_min, horizon=self.horizon,
                                                delta_d_max=config.sampling.d_max, delta_d_min=config.sampling.d_min)
        self.stopping_s = None

        # *****************************
        # Debug & Logger Initialization
        # *****************************
        self.log_risk = config.debug.log_risk
        self.save_all_traj = config.debug.save_all_traj
        self.all_traj = None
        self.optimal_trajectory = None
        self.use_occ_model = config.occlusion.use_occlusion_module
        self.logger = DataLoggingCosts(path_logs=log_path,
                                       save_all_traj=self.save_all_traj or self.use_amazing_visualizer,
                                       cost_params=config.cost.cost_weights)
        self._draw_traj_set = config.debug.draw_traj_set
        self._kinematic_debug = config.debug.kinematic_debug

        # **************************
        # Risk & Harm Initialization
        # **************************
        self.params_harm = load_harm_parameter_json(work_dir)
        self.params_risk = load_risk_json(work_dir)

        # **************************
        # Cost Function Setting
        # **************************
        cost_function = AdaptableCostFunction(rp=self, configuration=config)
        self.set_cost_function(cost_function=cost_function)

    @property
    def goal_checker(self):
        """Return the goal checker."""
        return self._goal_checker

    def _check_valid_settings(self):
        """Checks validity of provided dt and horizon"""
        assert is_positive(self.dT), '<ReactivePlanner>: provided dt is not correct! dt = {}'.format(self.dT)
        assert is_positive(self.N) and is_natural_number(self.N), '<ReactivePlanner>: N is not correct!'
        assert is_positive(self.horizon), '<ReactivePlanner>: provided t_h is not correct! dt = {}'.format(self.horizon)

    @property
    def collision_checker(self) -> pycrcc.CollisionChecker:
        return self._cc

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self._co

    @property
    def reference_path(self):
        return self._co.reference

    @property
    def infeasible_count_collision(self):
        """Number of colliding trajectories"""
        return self._infeasible_count_collision

    @property
    def infeasible_count_kinematics(self):
        """Number of kinematically infeasible trajectories"""
        return self._infeasible_count_kinematics

    def update_externals(self, scenario: Scenario = None, reference_path: np.ndarray = None,
                         planning_problem: PlanningProblem = None, goal_area: GoalRegion = None,
                         x_0: ReactivePlannerState = None, x_cl: Optional[Tuple[List, List]] = None,
                         cost_function=None, occlusion_module=None, desired_velocity: float = None,
                         predictions=None, reach_set=None, behavior=None):
        """
        Sets all external information in reactive planner
        :param scenario: Commonroad scenario
        :param reference_path: reference path as polyline
        :param planning_problem: reference path as polyline
        :param goal_area: commonroad goal area
        :param x_0: current ego vehicle state in global coordinate system
        :param x_cl: current ego vehicle state in curvilinear coordinate system
        :param cost_function: current used cost function
        :param occlusion_module: occlusion module setup
        :param desired_velocity: desired velocity in mps
        :param predictions: external calculated predictions of other obstacles
        :param reach_set: external calculated reach_sets
        :param behavior: behavior planner setup
        """
        if scenario is not None:
            self.set_scenario(scenario)
        if reference_path is not None:
            self.set_reference_path(reference_path)
        if planning_problem is not None:
            self.set_planning_problem(planning_problem)
        if goal_area is not None:
            self.set_goal_area(goal_area)
        if x_0 is not None:
            self.set_x_0(x_0)
            self.set_x_cl(x_cl)
        if cost_function is not None:
            self.set_cost_function(cost_function)
        if occlusion_module is not None:
            self.set_occlusion_module(occlusion_module)
        if desired_velocity is not None:
            self.set_desired_velocity(desired_velocity, x_0.velocity, self.stopping_s)
        if predictions is not None:
            self.set_predictions(predictions)
        if reach_set is not None:
            self.set_reach_set(reach_set)
        if behavior is not None:
            self.set_behavior(behavior)

    def set_scenario(self, scenario: Scenario):
        """Update the scenario to synchronize between agents"""
        self.scenario = scenario
        self.set_collision_checker(scenario)
        try:
            (
                _,
                self.road_boundary,
            ) = create_road_boundary_obstacle(
                scenario=self.scenario,
                method="aligned_triangulation",
                axis=2,
            )
        except:
            raise RuntimeError("Road Boundary can not be created")

    def set_predictions(self, predictions: dict):
        self.predictions = predictions

    def set_reach_set(self, reach_set):
        self.reach_set = reach_set

    def set_x_0(self, x_0: ReactivePlannerState):
        # set Cartesian initial state
        self.x_0 = x_0
        if self.x_0.velocity < self._low_vel_mode_threshold:
            self._LOW_VEL_MODE = True
        else:
            self._LOW_VEL_MODE = False

    def set_x_cl(self, x_cl):
        # set curvilinear initial state
        if self.x_cl is not None and not self.set_new_ref_path:
            self.x_cl = x_cl
        else:
            self.x_cl = self._compute_initial_states(self.x_0)
            self.set_new_ref_path = False

    def set_ego_vehicle_state(self, current_ego_vehicle):
        self.ego_vehicle_history.append(current_ego_vehicle)

    def set_behavior(self, behavior):
        self.behavior = behavior

    def record_state_and_input(self, state: ReactivePlannerState):
        """
        Adds state to list of recorded states
        Adds control inputs to list of recorded inputs
        """
        # append state to state list
        self.record_state_list.append(state)

        # compute control inputs and append to input list
        if len(self.record_state_list) > 1:
            steering_angle_speed = (state.steering_angle - self.record_state_list[-2].steering_angle) / self.dT
        else:
            steering_angle_speed = 0.0

        input_state = InputState(time_step=state.time_step,
                                 acceleration=state.acceleration,
                                 steering_angle_speed=steering_angle_speed)
        self.record_input_list.append(input_state)

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function
        self.logger.set_logging_header(self.config.cost.cost_weights)

    def set_reference_path(self, reference_path: np.ndarray = None, coordinate_system: CoordinateSystem = None):
        """
        Automatically creates a curvilinear coordinate system from a given reference path or sets a given
        curvilinear coordinate system for the planner to use
        :param reference_path: reference path as polyline
        :param coordinate_system: given CoordinateSystem object which is used by the planner
        """
        if coordinate_system is None:
            assert reference_path is not None, '<set reference path>: Please provide a reference path OR a ' \
                                               'CoordinateSystem object to the planner.'
            self._co: CoordinateSystem = CoordinateSystem(reference_path)
        else:
            assert reference_path is None, '<set reference path>: Please provide a reference path OR a ' \
                                           'CoordinateSystem object to the planner.'
            self._co: CoordinateSystem = coordinate_system
            self.set_new_ref_path = True

    def set_goal_area(self, goal_area: GoalRegion):
        """
        Sets the planning problem
        :param goal_area: Goal Area of Planning Problem
        """
        self.goal_area = goal_area

    def set_occlusion_module(self, occ_module):
        self.occlusion_module = occ_module

    def set_planning_problem(self, planning_problem: PlanningProblem):
        """
        Sets the planning problem
        :param planning_problem: PlanningProblem
        """
        self.planning_problem = planning_problem

    def set_sampling_parameters(self, t_min: float, horizon: float, delta_d_min: float, delta_d_max: float):
        """
        Sets sample parameters of time horizon
        :param t_min: minimum of sampled time horizon
        :param horizon: sampled time horizon
        :param delta_d_min: min lateral sampling
        :param delta_d_max: max lateral sampling
        """
        self.sampling_handler.update_static_params(t_min, horizon, delta_d_min, delta_d_max)

    def set_stopping_point(self, stop_s_coordinate):
        """
        Sets sample parameters of time horizon
        :param t_min: minimum of sampled time horizon
        """
        self.stopping_s = stop_s_coordinate

    def set_desired_velocity(self, desired_velocity: float, current_speed: float = None, stopping: float = False,
                             v_limit: float = 36):
        """
        Sets desired velocity and calculates velocity for each sample
        :param desired_velocity: velocity in m/s
        :param current_speed: velocity in m/s
        :param stopping
        :param v_limit: limit velocity due to behavior planner in m/s
        :return: velocity in m/s
        """
        self._desired_speed = desired_velocity

        min_v = max(0.01, current_speed - 0.75 * self.vehicle_params.a_max * self.horizon)
        max_v = min(min(current_speed + (self.vehicle_params.a_max / 7.0) * self.horizon, v_limit),
                    self.vehicle_params.v_max)

        self.sampling_handler.set_v_sampling(min_v, max_v)

        msg_logger.info('Sampled interval of velocity: {} m/s - {} m/s'.format(min_v, max_v))

    def set_collision_checker(self, scenario: Scenario = None, collision_checker: pycrcc.CollisionChecker = None):
        """
        Sets the collision checker used by the planner using either of the two options:
        If a collision_checker object is passed, then it is used directly by the planner.
        If no collision checker object is passed, then a CommonRoad scenario must be provided from which the collision
        checker is created and set.
        :param scenario: CommonRoad Scenario object
        :param collision_checker: pycrcc.CollisionChecker object
        """
        # self.scenario = scenario
        if collision_checker is None:
            assert scenario is not None, '<ReactivePlanner.set collision checker>: Please provide a CommonRoad scenario OR a ' \
                                         'CollisionChecker object to the planner.'
            cc_scenario = pycrcc.CollisionChecker()
            for co in scenario.static_obstacles:
                obs = create_collision_object(co)
                cc_scenario.add_collision_object(obs)
            for co in scenario.dynamic_obstacles:
                tvo = create_collision_object(co)
                cc_scenario.add_collision_object(tvo)
            _, road_boundary_sg_obb = create_road_boundary_obstacle(scenario)
            cc_scenario.add_collision_object(road_boundary_sg_obb)
            self._cc: pycrcc.CollisionChecker = cc_scenario
        else:
            assert scenario is None, '<ReactivePlanner.set collision checker>: Please provide a CommonRoad scenario OR a ' \
                                     'CollisionChecker object to the planner.'
            self._cc: pycrcc.CollisionChecker = collision_checker

    def set_risk_costs(self, trajectory):

        ego_risk_dict, obst_risk_dict, ego_harm_dict, obst_harm_dict, ego_risk, obst_risk = calc_risk(
            traj=trajectory,
            ego_state=self.x_0,
            predictions=self.predictions,
            scenario=self.scenario,
            ego_id=24,
            vehicle_params=self.vehicle_params,
            road_boundary=self.road_boundary,
            params_harm=self.params_harm,
            params_risk=self.params_risk,
        )
        trajectory._ego_risk = ego_risk
        trajectory._obst_risk = obst_risk
        return trajectory

    def plan(self) -> tuple:
        """
        Plans an optimal trajectory
        :return: Optimal trajectory as tuple
        """

        msg_logger.debug('Initial state is: lon = {} / lat = {}'.format(self.x_cl[0], self.x_cl[1]))
        msg_logger.debug('Desired velocity is {} m/s'.format(self._desired_speed))

        # initialize optimal trajectory dummy
        optimal_trajectory = None
        trajectory_pair = None
        t0 = time.time()

        # initial index of sampling set to use
        i = self._sampling_min  # Time sampling is not used. To get more samples, start with level 1.

        # sample until trajectory has been found or sampling sets are empty
        while optimal_trajectory is None and i < self._sampling_max:

            self.cost_function.update_state(scenario=self.scenario, rp=self,
                                            predictions=self.predictions, reachset=self.reach_set)

            # TODO: Stopping Mode (set all traj beyond specified S coordinate to invalid)

            # bundle = self._create_end_point_trajectory_bundle(self.x_cl[0], self.x_cl[1], self.x_cl[0][0]+20, self.cost_function, samp_level=i)
            bundle = self._create_trajectory_bundle(self.x_cl[0], self.x_cl[1], self.cost_function, samp_level=i)

            self.logger.trajectory_number = self.x_0.time_step

            optimal_trajectory = self._get_optimal_trajectory(bundle, i)
            trajectory_pair = self._compute_trajectory_pair(optimal_trajectory) if optimal_trajectory is not None else None

            # create CommonRoad Obstacle for the ego Vehicle
            if trajectory_pair is not None:
                current_ego_vehicle = self.convert_state_list_to_commonroad_object(trajectory_pair[0].state_list)
                self.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

            if optimal_trajectory is not None and self.log_risk:
                optimal_trajectory = self.set_risk_costs(optimal_trajectory)

            if self.behavior:
                if self.behavior.flags["waiting_for_green_light"]:
                    optimal_trajectory = self._compute_standstill_trajectory()

            msg_logger.debug('Rejected {} infeasible trajectories due to kinematics'.format(
                self.infeasible_count_kinematics))
            msg_logger.debug('Rejected {} infeasible trajectories due to collisions'.format(
                self.infeasible_count_collision))

            # increase sampling level (i.e., density) if no optimal trajectory could be found
            i = i + 1

        planning_time = time.time() - t0
        if optimal_trajectory is None and self.x_0.velocity <= 0.1:
            msg_logger.warning('Planning standstill for the current scenario')
            self.logger.trajectory_number = self.x_0.time_step
            optimal_trajectory = self._compute_standstill_trajectory()

        # **************************
        # Logging
        # **************************
        if optimal_trajectory is not None:
            self.logger.log(optimal_trajectory, infeasible_kinematics=self.infeasible_count_kinematics,
                            percentage_kinematics=self.infeasible_kinematics_percentage,
                            infeasible_collision=self.infeasible_count_collision, planning_time=planning_time,
                            ego_vehicle=self.ego_vehicle_history[-1])
            self.logger.log_predicition(self.predictions)
        if self.save_all_traj or self.use_amazing_visualizer:
            self.logger.log_all_trajectories(self.all_traj, self.x_0.time_step)

        # **************************
        # Check Cost Status
        # **************************
        if optimal_trajectory is not None and self.x_0.time_step > 0:
            self._optimal_cost = optimal_trajectory.cost
            msg_logger.debug('Found optimal trajectory with {}% of maximum seen costs'
                  .format(int((self._optimal_cost/self.max_seen_costs)*100)))

        if optimal_trajectory is not None:
            if self.max_seen_costs < self._optimal_cost:
                self.max_seen_costs = self._optimal_cost

        return trajectory_pair

    def _compute_initial_states(self, x_0: ReactivePlannerState) -> (np.ndarray, np.ndarray):
        """
        Computes the curvilinear initial states for the polynomial planner based on a Cartesian CommonRoad state
        :param x_0: The CommonRoad state object representing the initial state of the vehicle
        :return: A tuple containing the initial longitudinal and lateral states (lon,lat)
        """
        # compute curvilinear position
        try:
            s, d = self._co.convert_to_curvilinear_coords(x_0.position[0], x_0.position[1])
        except ValueError:
            msg_logger.critical("Initial state could not be transformed.")
            raise ValueError("Initial state could not be transformed.")

        # factor for interpolation
        s_idx = np.argmax(self._co.ref_pos > s) - 1
        s_lambda = (s - self._co.ref_pos[s_idx]) / (
                self._co.ref_pos[s_idx + 1] - self._co.ref_pos[s_idx])

        # compute orientation in curvilinear coordinate frame
        ref_theta = np.unwrap(self._co.ref_theta)
        theta_cl = x_0.orientation - interpolate_angle(s, self._co.ref_pos[s_idx], self._co.ref_pos[s_idx + 1],
                                                       ref_theta[s_idx], ref_theta[s_idx + 1])

        # compute reference curvature
        kr = (self._co.ref_curv[s_idx + 1] - self._co.ref_curv[s_idx]) * s_lambda + self._co.ref_curv[
            s_idx]
        # compute reference curvature change
        kr_d = (self._co.ref_curv_d[s_idx + 1] - self._co.ref_curv_d[s_idx]) * s_lambda + self._co.ref_curv_d[s_idx]

        # compute initial ego curvature from initial steering angle
        kappa_0 = np.tan(x_0.steering_angle) / self.vehicle_params.wheelbase

        # compute d' and d'' -> derivation after arclength (s): see Eq. (A.3) and (A.5) in Diss. Werling
        d_p = (1 - kr * d) * np.tan(theta_cl)
        d_pp = -(kr_d * d + kr * d_p) * np.tan(theta_cl) + ((1 - kr * d) / (math.cos(theta_cl) ** 2)) * (
                kappa_0 * (1 - kr * d) / math.cos(theta_cl) - kr)

        # compute s dot (s_velocity) and s dot dot (s_acceleration) -> derivation after time
        s_velocity = x_0.velocity * math.cos(theta_cl) / (1 - kr * d)
        if s_velocity < 0:
            raise Exception("Initial state or reference incorrect! Curvilinear velocity is negative which indicates"
                            "that the ego vehicle is not driving in the same direction as specified by the reference")

        s_acceleration = x_0.acceleration
        s_acceleration -= (s_velocity ** 2 / math.cos(theta_cl)) * (
                (1 - kr * d) * np.tan(theta_cl) * (kappa_0 * (1 - kr * d) / (math.cos(theta_cl)) - kr) -
                (kr_d * d + kr * d_p))
        s_acceleration /= ((1 - kr * d) / (math.cos(theta_cl)))

        # compute d dot (d_velocity) and d dot dot (d_acceleration)
        if self._LOW_VEL_MODE:
            # in LOW_VEL_MODE: d_velocity and d_acceleration are derivatives w.r.t arclength (s)
            d_velocity = d_p
            d_acceleration = d_pp
        else:
            # in HIGH VEL MODE: d_velocity and d_acceleration are derivatives w.r.t time
            d_velocity = x_0.velocity * math.sin(theta_cl)
            d_acceleration = s_acceleration * d_p + s_velocity ** 2 * d_pp

        x_0_lon: List[float] = [s, s_velocity, s_acceleration]
        x_0_lat: List[float] = [d, d_velocity, d_acceleration]

        msg_logger.debug(f'Initial state for planning is {x_0}')
        msg_logger.debug(f'Initial x_0 lon = {x_0_lon}')
        msg_logger.debug(f'Initial x_0 lat = {x_0_lat}')

        return x_0_lon, x_0_lat

    def _create_trajectory_bundle(self, x_0_lon: np.array, x_0_lat: np.array, cost_function, samp_level: int) -> TrajectoryBundle:
        """
        Plans trajectory samples that try to reach a certain velocity and samples in this domain.
        Sample in time (duration) and velocity domain. Initial state is given. Longitudinal end state (s) is sampled.
        Lateral end state (d) is always set to 0.
        :param x_0_lon: np.array([s, s_dot, s_ddot])
        :param x_0_lat: np.array([d, d_dot, d_ddot])
        :param samp_level: index of the sampling parameter set to use
        :return: trajectory bundle with all sample trajectories.

        NOTE: Here, no collision or feasibility check is done!
        """
        # reset cost statistic
        self._min_cost = 10 ** 9
        self._max_cost = 0

        trajectories = list()
        for t in self.sampling_handler.t_sampling.to_range(samp_level):
            # Longitudinal sampling for all possible velocities
            for v in self.sampling_handler.v_sampling.to_range(samp_level):
                # end_state_lon = np.array([t * v + x_0_lon[0], v, 0.0])
                # trajectory_long = QuinticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lon), x_d=end_state_lon)
                trajectory_long = QuarticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lon), x_d=np.array([v, 0]))

                # Sample lateral end states (add x_0_lat to sampled states)
                if trajectory_long.coeffs is not None:
                    for d in self.sampling_handler.d_sampling.to_range(samp_level).union({x_0_lat[0]}):
                        end_state_lat = np.array([d, 0.0, 0.0])
                        # SWITCHING TO POSITION DOMAIN FOR LATERAL TRAJECTORY PLANNING
                        if self._LOW_VEL_MODE:
                            s_lon_goal = trajectory_long.evaluate_state_at_tau(t)[0] - x_0_lon[0]
                            if s_lon_goal <= 0:
                                s_lon_goal = t
                            trajectory_lat = QuinticTrajectory(tau_0=0, delta_tau=s_lon_goal, x_0=np.array(x_0_lat),
                                                               x_d=end_state_lat)

                        # Switch to sampling over t for high velocities
                        else:
                            trajectory_lat = QuinticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lat),
                                                               x_d=end_state_lat)
                        if trajectory_lat.coeffs is not None:
                            trajectory_sample = TrajectorySample(self.horizon, self.dT, trajectory_long, trajectory_lat,
                                                                 len(trajectories), costMap=self.cost_function.cost_weights)
                            trajectories.append(trajectory_sample)

        # perform pre-check and order trajectories according their cost
        trajectory_bundle = TrajectoryBundle(trajectories, cost_function=cost_function,
                                             multiproc=self._multiproc, num_workers=self._num_workers)
        self._total_count = len(trajectory_bundle._trajectory_bundle)
        msg_logger.debug('%s trajectories sampled' % len(trajectory_bundle._trajectory_bundle))
        return trajectory_bundle

    def _get_optimal_trajectory(self, trajectory_bundle: TrajectoryBundle, samp_lvl):
        """
        Computes the optimal trajectory from a given trajectory bundle
        :param trajectory_bundle: The trajectory bundle
        :return: The optimal trajectory if exists (otherwise None)
        """
        # VALIDITY_LEVELS = {
        #     0: "Physically invalid",
        #     1: "Collision",
        #     2: "Leaving road boundaries",
        #     3: "Maximum acceptable risk",
        #     10: "Valid",
        # }
        # reset statistics
        self._infeasible_count_collision = 0
        self._infeasible_count_kinematics = np.zeros(10)

        # check kinematics of each trajectory
        if self._multiproc:
            # with multiprocessing
            # divide trajectory_bundle.trajectories into chunks
            chunk_size = math.ceil(len(trajectory_bundle.trajectories) / self._num_workers)
            chunks = [trajectory_bundle.trajectories[ii * chunk_size: min(len(trajectory_bundle.trajectories),
                                                                          (ii+1)*chunk_size)] for ii in range(0, self._num_workers)]

            # initialize list of Processes and Queues
            list_processes = []
            feasible_trajectories = []
            queue_1 = multiprocessing.Queue()
            infeasible_trajectories = []
            queue_2 = multiprocessing.Queue()
            infeasible_count_kinematics = [0] * 10
            queue_3 = multiprocessing.Queue()
            for chunk in chunks:
                p = Process(target=self.check_feasibility, args=(chunk, queue_1, queue_2, queue_3))
                list_processes.append(p)
                p.start()

            # get return values from queue
            for p in list_processes:
                feasible_trajectories.extend(queue_1.get())
                if self._draw_traj_set:
                    infeasible_trajectories.extend(queue_2.get())
                if self._kinematic_debug:
                    temp = queue_3.get()
                    infeasible_count_kinematics = [x + y for x, y in zip(infeasible_count_kinematics, temp)]

            # wait for all processes to finish
            for p in list_processes:
                p.join()
        else:
            # without multiprocessing
            feasible_trajectories, infeasible_trajectories, infeasible_count_kinematics = \
                                                            self.check_feasibility(trajectory_bundle.trajectories)

        if self.use_occ_model and feasible_trajectories:
            self.occlusion_module.occ_phantom_module.evaluate_trajectories(feasible_trajectories)
            # self.occlusion_module.occ_visibility_estimator.evaluate_trajectories(feasible_trajectories, predictions)
            self.occlusion_module.occ_uncertainty_map_evaluator.evaluate_trajectories(feasible_trajectories)

        msg_logger.debug('Kinematic check of %s trajectories done' % len(trajectory_bundle.trajectories))

        # update number of infeasible trajectories
        self._infeasible_count_kinematics = infeasible_count_kinematics
        self._infeasible_count_kinematics[0] = len(trajectory_bundle.trajectories) - len(feasible_trajectories)
        self.infeasible_kinematics_percentage = float(len(feasible_trajectories)/
                                                      len(trajectory_bundle.trajectories)) * 100
        # print(self.infeasible_kinematics_percentage)
        # for visualization store all trajectories with validity level based on kinematic validity
        if self._draw_traj_set or self.save_all_traj or self.use_amazing_visualizer:
            for traj in feasible_trajectories:
                setattr(traj, 'feasible', True)
            for traj in infeasible_trajectories:
                setattr(traj, 'feasible', False)
            trajectory_bundle.trajectories = feasible_trajectories + infeasible_trajectories
            trajectory_bundle.sort(occlusion_module=self.occlusion_module)
            self.all_traj = trajectory_bundle.trajectories
            trajectory_bundle.trajectories = list(filter(lambda x: x.feasible is True, trajectory_bundle.trajectories))
        else:
            # set feasible trajectories in bundle
            trajectory_bundle.trajectories = feasible_trajectories
            # sort trajectories according to their costs
            trajectory_bundle.sort(occlusion_module=self.occlusion_module)

        # ******************************************
        # Check Feasible Trajectories for Collisions
        # ******************************************
        optimal_trajectory = self.trajectory_collision_check(feasible_trajectories=
                                                             trajectory_bundle.get_sorted_list(
                                                                 occlusion_module=self.occlusion_module))

        if samp_lvl >= self._sampling_max - 1 and optimal_trajectory is None and feasible_trajectories:
            for traje in feasible_trajectories:
                self.set_risk_costs(traje)
            sort_risk = sorted(feasible_trajectories, key=lambda traj: traj._ego_risk + traj._obst_risk,
                               reverse=False)
            optimal_trajectory = sort_risk[0]
            return optimal_trajectory

        else:
            return optimal_trajectory

    def check_feasibility(self, trajectories: List[TrajectorySample], queue_1=None, queue_2=None, queue_3=None):
        """
        Checks the kinematics of given trajectories in a bundle and computes the cartesian trajectory information
        Lazy evaluation, only kinematically feasible trajectories are evaluated further

        :param trajectories: The list of trajectory samples to check
        :param queue_1: Multiprocessing.Queue() object for storing feasible trajectories
        :param queue_2: Multiprocessing.Queue() object for storing infeasible trajectories (only vor visualization)
        :param queue_3: Multiprocessing.Queue() object for storing reason for infeasible trajectory in list
        :return: The list of output trajectories
        """
        # initialize lists for output trajectories
        # infeasible trajectory list is only used for visualization when self._draw_traj_set is True
        infeasible_count_kinematics = np.zeros(10)
        feasible_trajectories = list()
        infeasible_trajectories = list()

        # loop over list of trajectories
        for trajectory in trajectories:
            # create time array and precompute time interval information
            t = np.arange(0, np.round(trajectory.trajectory_long.delta_tau + trajectory.dt, 5), trajectory.dt)
            t2 = np.round(np.power(t, 2), 10)
            t3 = np.round(np.power(t, 3), 10)
            t4 = np.round(np.power(t, 4), 10)
            t5 = np.round(np.power(t, 5), 10)

            # length of the trajectory sample (i.e., number of time steps. can be smaller than planning horizon)
            traj_len = len(t)

            # initialize long. (s) and lat. (d) state vectors
            s = np.zeros(self.N + 1)
            s_velocity = np.zeros(self.N + 1)
            s_acceleration = np.zeros(self.N + 1)
            d = np.zeros(self.N + 1)
            d_velocity = np.zeros(self.N + 1)
            d_acceleration = np.zeros(self.N + 1)

            # compute longitudinal position, velocity, acceleration from trajectory sample
            s[:traj_len] = trajectory.trajectory_long.calc_position(t, t2, t3, t4, t5)  # lon pos
            s_velocity[:traj_len] = trajectory.trajectory_long.calc_velocity(t, t2, t3, t4)  # lon velocity
            s_acceleration[:traj_len] = trajectory.trajectory_long.calc_acceleration(t, t2, t3)  # lon acceleration

            # At low speeds, we have to sample the lateral motion over the travelled distance rather than time.
            if not self._LOW_VEL_MODE:
                d[:traj_len] = trajectory.trajectory_lat.calc_position(t, t2, t3, t4, t5)  # lat pos
                d_velocity[:traj_len] = trajectory.trajectory_lat.calc_velocity(t, t2, t3, t4)  # lat velocity
                d_acceleration[:traj_len] = trajectory.trajectory_lat.calc_acceleration(t, t2, t3)  # lat acceleration
            else:
                # compute normalized travelled distance for low velocity mode of lateral planning
                s1 = s[:traj_len] - s[0]
                s2 = np.square(s1)
                s3 = s2 * s1
                s4 = np.square(s2)
                s5 = s4 * s1

                # compute lateral position, velocity, acceleration from trajectory sample
                d[:traj_len] = trajectory.trajectory_lat.calc_position(s1, s2, s3, s4, s5)  # lat pos
                # in LOW_VEL_MODE d_velocity is actually d' (see Diss. Moritz Werling  p.124)
                d_velocity[:traj_len] = trajectory.trajectory_lat.calc_velocity(s1, s2, s3, s4)  # lat velocity
                d_acceleration[:traj_len] = trajectory.trajectory_lat.calc_acceleration(s1, s2, s3)  # lat acceleration

            # precision for near zero velocities from evaluation of polynomial coefficients
            # set small velocities to zero
            s_velocity[np.abs(s_velocity) < _EPS] = 0.0
            d_velocity[np.abs(d_velocity) < _EPS] = 0.0

            # Initialize trajectory state vectors
            # (Global) Cartesian positions x, y
            x = np.zeros(self.N + 1)
            y = np.zeros(self.N + 1)
            # (Global) Cartesian velocity v and acceleration a
            v = np.zeros(self.N + 1)
            a = np.zeros(self.N + 1)
            # Orientation theta: Cartesian (gl) and Curvilinear (cl)
            theta_gl = np.zeros(self.N + 1)
            theta_cl = np.zeros(self.N + 1)
            # Curvature kappa : Cartesian (gl) and Curvilinear (cl)
            kappa_gl = np.zeros(self.N + 1)
            kappa_cl = np.zeros(self.N + 1)

            # Initialize Feasibility boolean
            feasible = True

            if not self._draw_traj_set:
                # pre-filter with quick underapproximative check for feasibility
                if np.any(np.abs(s_acceleration) > self.vehicle_params.a_max):
                    msg_logger.debug(f"Acceleration {np.max(np.abs(s_acceleration))}")
                    feasible = False
                    infeasible_count_kinematics[1] += 1
                    infeasible_trajectories.append(trajectory)
                    continue
                if np.any(s_velocity < -_EPS):
                    msg_logger.debug(f"Velocity {min(s_velocity)} at step")
                    feasible = False
                    infeasible_count_kinematics[2] += 1
                    infeasible_trajectories.append(trajectory)
                    continue

            infeasible_count_kinematics_traj = np.zeros(10)
            for i in range(0, traj_len):
                # compute orientations
                # see Appendix A.1 of Moritz Werling's PhD Thesis for equations
                if not self._LOW_VEL_MODE:
                    if s_velocity[i] > 0.001:
                        dp = d_velocity[i] / s_velocity[i]
                    else:
                        # if abs(d_velocity[i]) > 0.001:
                        #     dp = None
                        # else:
                        dp = 0.
                    # see Eq. (A.8) from Moritz Werling's Diss
                    ddot = d_acceleration[i] - dp * s_acceleration[i]

                    if s_velocity[i] > 0.001:
                        dpp = ddot / (s_velocity[i] ** 2)
                    else:
                        # if np.abs(ddot) > 0.00003:
                        #     dpp = None
                        # else:
                        dpp = 0.
                else:
                    dp = d_velocity[i]
                    dpp = d_acceleration[i]

                # factor for interpolation
                s_idx = np.argmax(self._co.ref_pos > s[i]) - 1
                if s_idx + 1 >= len(self._co.ref_pos):
                    feasible = False
                    infeasible_count_kinematics_traj[3] = 1
                    break
                s_lambda = (s[i] - self._co.ref_pos[s_idx]) / (self._co.ref_pos[s_idx + 1] - self._co.ref_pos[s_idx])

                # compute curvilinear (theta_cl) and global Cartesian (theta_gl) orientation
                if s_velocity[i] > 0.001:
                    # LOW VELOCITY MODE: dp = d_velocity[i]
                    # HIGH VELOCITY MODE: dp = d_velocity[i]/s_velocity[i]
                    theta_cl[i] = np.arctan2(dp, 1.0)

                    theta_gl[i] = theta_cl[i] + interpolate_angle(
                        s[i],
                        self._co.ref_pos[s_idx],
                        self._co.ref_pos[s_idx + 1],
                        self._co.ref_theta[s_idx],
                        self._co.ref_theta[s_idx + 1])
                else:
                    if self._LOW_VEL_MODE:
                        # dp = velocity w.r.t. to travelled arclength (s)
                        theta_cl[i] = np.arctan2(dp, 1.0)

                        theta_gl[i] = theta_cl[i] + interpolate_angle(
                            s[i],
                            self._co.ref_pos[s_idx],
                            self._co.ref_pos[s_idx + 1],
                            self._co.ref_theta[s_idx],
                            self._co.ref_theta[s_idx + 1])
                    else:
                        # in stillstand (s_velocity~0) and High velocity mode: assume vehicle keeps global orientation
                        theta_gl[i] = self.x_0.orientation if i == 0 else theta_gl[i - 1]

                        theta_cl[i] = theta_gl[i] - interpolate_angle(
                            s[i],
                            self._co.ref_pos[s_idx],
                            self._co.ref_pos[s_idx + 1],
                            self._co.ref_theta[s_idx],
                            self._co.ref_theta[s_idx + 1])

                # Interpolate curvature of reference path k_r at current position
                k_r = (self._co.ref_curv[s_idx + 1] - self._co.ref_curv[s_idx]) * s_lambda + self._co.ref_curv[s_idx]
                # Interpolate curvature rate of reference path k_r_d at current position
                k_r_d = (self._co.ref_curv_d[s_idx + 1] - self._co.ref_curv_d[s_idx]) * s_lambda + \
                        self._co.ref_curv_d[s_idx]

                # compute global curvature (see appendix A of Moritz Werling's PhD thesis)
                oneKrD = (1 - k_r * d[i])
                cosTheta = math.cos(theta_cl[i])
                tanTheta = np.tan(theta_cl[i])

                kappa_gl[i] = (dpp + (k_r * dp + k_r_d * d[i]) * tanTheta) * cosTheta * ((cosTheta / oneKrD) ** 2) + \
                              (cosTheta / oneKrD) * k_r

                kappa_cl[i] = kappa_gl[i] - k_r

                # compute (global) Cartesian velocity
                v[i] = s_velocity[i] * (oneKrD / (math.cos(theta_cl[i])))

                # compute (global) Cartesian acceleration
                a[i] = s_acceleration[i] * (oneKrD / cosTheta) + ((s_velocity[i] ** 2) / cosTheta) * (
                        oneKrD * tanTheta * (kappa_gl[i] * (oneKrD / cosTheta) - k_r) - (
                        k_r_d * d[i] + k_r * dp))

                # **************************
                # Velocity constraint
                # **************************
                if v[i] < -_EPS:
                    feasible = False
                    infeasible_count_kinematics_traj[4] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

                # **************************
                # Curvature constraint
                # **************************
                kappa_max = np.tan(self.vehicle_params.delta_max) / self.vehicle_params.wheelbase
                if abs(kappa_gl[i]) > kappa_max:
                    feasible = False
                    infeasible_count_kinematics_traj[5] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

                # **************************
                # Yaw rate constraint
                # **************************
                yaw_rate = (theta_gl[i] - theta_gl[i - 1]) / self.dT if i > 0 else 0.
                theta_dot_max = kappa_max * v[i]
                if abs(round(yaw_rate, 5)) > theta_dot_max:
                    feasible = False
                    infeasible_count_kinematics_traj[6] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

                # **************************
                # Curvature rate constraint
                # **************************
                # steering_angle = np.arctan2(self.vehicle_params.wheelbase * kappa_gl[i], 1.0)
                # kappa_dot_max = self.vehicle_params.v_delta_max / (self.vehicle_params.wheelbase *
                #                                                    math.cos(steering_angle) ** 2)
                kappa_dot = (kappa_gl[i] - kappa_gl[i - 1]) / self.dT if i > 0 else 0.
                if abs(kappa_dot) > 0.4:
                    feasible = False
                    infeasible_count_kinematics_traj[7] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

                # **************************
                # Acceleration rate constraint
                # **************************
                v_switch = self.vehicle_params.v_switch
                a_max = self.vehicle_params.a_max * v_switch / v[i] if v[i] > v_switch else self.vehicle_params.a_max
                a_min = -self.vehicle_params.a_max
                if not a_min <= a[i] <= a_max:
                    feasible = False
                    infeasible_count_kinematics_traj[8] = 1
                    if not self._draw_traj_set and not self._kinematic_debug:
                        break

            # if selected polynomial trajectory is feasible, store it's Cartesian and Curvilinear trajectory
            if feasible or self._draw_traj_set:
                # Extend Trajectory to get same lenth
                t_ext = np.arange(1, len(s) - traj_len + 1, 1) * trajectory.dt
                s[traj_len:] = s[traj_len-1] + t_ext * s_velocity[traj_len-1]
                d[traj_len:] = d[traj_len-1]
                for i in range(0, len(s)):
                    # compute (global) Cartesian position
                    pos: np.ndarray = self._co.convert_to_cartesian_coords(s[i], d[i])
                    if pos is not None:
                        x[i] = pos[0]
                        y[i] = pos[1]
                    else:
                        feasible = False
                        infeasible_count_kinematics_traj[9] = 1
                        msg_logger.debug("Out of projection domain")
                        break

                if feasible or self._draw_traj_set:
                    # store Cartesian trajectory
                    trajectory.cartesian = CartesianSample(x, y, theta_gl, v, a, kappa_gl,
                                                           kappa_dot=np.append([0], np.diff(kappa_gl)),
                                                           current_time_step=traj_len)

                    # store Curvilinear trajectory
                    trajectory.curvilinear = CurviLinearSample(s, d, theta_cl,
                                                               ss=s_velocity, sss=s_acceleration,
                                                               dd=d_velocity, ddd=d_acceleration,
                                                               current_time_step=traj_len)

                    trajectory.actual_traj_length = traj_len

                    # check if trajectories planning horizon is shorter than expected and extend if necessary
                    if self.N + 1 > trajectory.cartesian.current_time_step:
                        trajectory.enlarge(self.dT)

                if feasible:
                    feasible_trajectories.append(trajectory)
                elif not feasible and self._draw_traj_set:
                    infeasible_trajectories.append(trajectory)

            infeasible_count_kinematics += infeasible_count_kinematics_traj

        if self._multiproc:
            # store feasible trajectories in Queue 1
            queue_1.put(feasible_trajectories)
            # if visualization is required: store infeasible trajectories in Queue 1
            if self._draw_traj_set:
                queue_2.put(infeasible_trajectories)
            if self._kinematic_debug:
                queue_3.put(infeasible_count_kinematics)
        else:
            return feasible_trajectories, infeasible_trajectories, infeasible_count_kinematics

    def trajectory_collision_check(self, feasible_trajectories):
        """
        Checks feasible trajectories for collisions with static obstacles
        :param feasible_trajectories: feasible trajectories list
        :return trajectory: optimal feasible trajectory or None
        """
        # go through sorted list of sorted trajectories and check for collisions
        for trajectory in feasible_trajectories:
            # Add Occupancy of Trajectory to do Collision Checks later
            cart_traj = self._compute_cart_traj(trajectory)
            occupancy = self.convert_state_list_to_commonroad_object(cart_traj.state_list)
            # get collision_object
            coll_obj = self.create_coll_object(occupancy, self.vehicle_params, self.x_0)

            # TODO: Check kinematic checks in cpp. no feasible traj available
            if self.use_prediction:
                collision_detected = collision_checker_prediction(
                    predictions=self.predictions,
                    scenario=self.scenario,
                    ego_co=coll_obj,
                    frenet_traj=trajectory,
                    ego_state=self.x_0,
                )
                if collision_detected:
                    self._infeasible_count_collision += 1
            else:
                collision_detected = False

            leaving_road_at = trajectories_collision_static_obstacles(
                trajectories=[coll_obj],
                static_obstacles=self.road_boundary,
                method="grid",
                num_cells=32,
                auto_orientation=True,
            )
            if leaving_road_at[0] != -1:
                coll_time_step = leaving_road_at[0] - self.x_0.time_step
                coll_vel = trajectory.cartesian.v[coll_time_step]

                boundary_harm = get_protected_inj_prob_log_reg_ignore_angle(
                    velocity=coll_vel, coeff=self.params_harm
                )

            else:
                boundary_harm = 0

            # Save Status of Trajectory to sort for alternative
            trajectory.boundary_harm = boundary_harm
            trajectory._coll_detected = collision_detected

            if not collision_detected and boundary_harm == 0:
                return trajectory

        return None

    def _compute_trajectory_pair(self, trajectory: TrajectorySample) -> tuple:
        """
        Computes the output required for visualizing in CommonRoad framework
        :param trajectory: the optimal trajectory
        :return: (CartesianTrajectory, FrenetTrajectory, lon sample, lat sample)
        """
        # go along state list
        cart_list = list()
        cl_list = list()

        lon_list = list()
        lat_list = list()
        for i in range(len(trajectory.cartesian.x)):
            # create Cartesian state
            cart_states = dict()
            cart_states['time_step'] = self.x_0.time_step+i
            cart_states['position'] = np.array([trajectory.cartesian.x[i], trajectory.cartesian.y[i]])
            cart_states['orientation'] = trajectory.cartesian.theta[i]
            cart_states['velocity'] = trajectory.cartesian.v[i]
            cart_states['acceleration'] = trajectory.cartesian.a[i]
            if i > 0:
                cart_states['yaw_rate'] = (trajectory.cartesian.theta[i] - trajectory.cartesian.theta[i-1]) / self.dT
            else:
                cart_states['yaw_rate'] = self.x_0.yaw_rate
            # TODO Check why computation with yaw rate was faulty ??
            cart_states['steering_angle'] = np.arctan2(self.vehicle_params.wheelbase *
                                                       trajectory.cartesian.kappa[i], 1.0)
            cart_list.append(ReactivePlannerState(**cart_states))

            # create curvilinear state
            # TODO: This is not correct
            cl_states = dict()
            cl_states['time_step'] = self.x_0.time_step+i
            cl_states['position'] = np.array([trajectory.curvilinear.s[i], trajectory.curvilinear.d[i]])
            cl_states['velocity'] = trajectory.cartesian.v[i]
            cl_states['acceleration'] = trajectory.cartesian.a[i]
            cl_states['orientation'] = trajectory.cartesian.theta[i]
            cl_states['yaw_rate'] = trajectory.cartesian.kappa[i]
            cl_list.append(CustomState(**cl_states))

            lon_list.append(
                [trajectory.curvilinear.s[i], trajectory.curvilinear.s_dot[i], trajectory.curvilinear.s_ddot[i]])
            lat_list.append(
                [trajectory.curvilinear.d[i], trajectory.curvilinear.d_dot[i], trajectory.curvilinear.d_ddot[i]])

        # make Cartesian and Curvilinear Trajectory
        cartTraj = Trajectory(self.x_0.time_step, cart_list)
        cvlnTraj = Trajectory(self.x_0.time_step, cl_list)

        # correct orientations of cartesian output trajectory
        cartTraj_corrected = self.shift_orientation(cartTraj, interval_start=self.x_0.orientation - np.pi,
                                                    interval_end=self.x_0.orientation + np.pi)

        return cartTraj_corrected, cvlnTraj, lon_list, lat_list

    def _compute_cart_traj(self, trajectory: TrajectorySample) -> Trajectory:
        """
        Computes the output required for visualizing in CommonRoad framework
        :param trajectory: the optimal trajectory
        :return: (CartesianTrajectory, FrenetTrajectory, lon sample, lat sample)
        """
        # go along state list
        cart_list = list()

        for i in range(len(trajectory.cartesian.x)):
            # create Cartesian state
            cart_states = dict()
            cart_states['time_step'] = self.x_0.time_step+i
            cart_states['position'] = np.array([trajectory.cartesian.x[i], trajectory.cartesian.y[i]])
            cart_states['orientation'] = trajectory.cartesian.theta[i]
            cart_states['velocity'] = trajectory.cartesian.v[i]
            cart_states['acceleration'] = trajectory.cartesian.a[i]
            if i > 0:
                cart_states['yaw_rate'] = (trajectory.cartesian.theta[i] - trajectory.cartesian.theta[i-1]) / self.dT
            else:
                cart_states['yaw_rate'] = self.x_0.yaw_rate
            # TODO Check why computation with yaw rate was faulty ??
            cart_states['steering_angle'] = np.arctan2(self.vehicle_params.wheelbase *
                                                       trajectory.cartesian.kappa[i], 1.0)
            cart_list.append(ReactivePlannerState(**cart_states))

        # make Cartesian and Curvilinear Trajectory
        cartTraj = Trajectory(self.x_0.time_step, cart_list)

        return cartTraj

    def _compute_standstill_trajectory(self) -> TrajectorySample:
        """
        Computes a standstill trajectory if the vehicle is already at velocity 0
        :return: The TrajectorySample for a standstill trajectory
        """
        # current planner initial state
        x_0 = self.x_0
        x_0_lon, x_0_lat = self.x_cl

        # create artificial standstill trajectory
        msg_logger.debug('Adding standstill trajectory')
        msg_logger.debug("x_0 is {}".format(x_0))
        msg_logger.debug("x_0_lon is {}".format(x_0_lon))
        msg_logger.debug("x_0_lon is {}".format(type(x_0_lon)))

        # create lon and lat polynomial
        traj_lon = QuarticTrajectory(tau_0=0, delta_tau=self.horizon, x_0=np.asarray(x_0_lon),
                                     x_d=np.array([0, 0]))
        traj_lat = QuinticTrajectory(tau_0=0, delta_tau=self.horizon, x_0=np.asarray(x_0_lat),
                                     x_d=np.array([x_0_lat[0], 0, 0]))

        # compute initial ego curvature (global coordinates) from initial steering angle
        kappa_0 = np.tan(x_0.steering_angle) / self.vehicle_params.wheelbase

        # create Trajectory sample
        p = TrajectorySample(self.horizon, self.dT, traj_lon, traj_lat, uniqueId=0,
                             costMap=self.cost_function.cost_weights)

        # create Cartesian trajectory sample
        a = np.repeat(0.0, self.N)
        a[1] = - self.x_0.velocity / self.dT
        p.cartesian = CartesianSample(np.repeat(x_0.position[0], self.N), np.repeat(x_0.position[1], self.N),
                                      np.repeat(x_0.orientation, self.N), np.repeat(0.0, self.N),
                                      a, np.repeat(kappa_0, self.N), np.repeat(0.0, self.N),
                                      current_time_step=self.N)

        # create Curvilinear trajectory sample
        # compute orientation in curvilinear coordinate frame
        s_idx = np.argmax(self._co.ref_pos > x_0_lon[0]) - 1
        ref_theta = np.unwrap(self._co.ref_theta)
        theta_cl = x_0.orientation - interpolate_angle(x_0_lon[0], self._co.ref_pos[s_idx], self._co.ref_pos[s_idx + 1],
                                                       ref_theta[s_idx], ref_theta[s_idx + 1])

        p.curvilinear = CurviLinearSample(np.repeat(x_0_lon[0], self.N), np.repeat(x_0_lat[0], self.N),
                                          np.repeat(theta_cl, self.N), dd=np.repeat(x_0_lat[1], self.N),
                                          ddd=np.repeat(x_0_lat[2], self.N), ss=np.repeat(x_0_lon[1], self.N),
                                          sss=np.repeat(x_0_lon[2], self.N), current_time_step=self.N)
        return p

    def _create_end_point_trajectory_bundle(self, x_0_lon, x_0_lat, stop_point_s, cost_function, samp_level):
        msg_logger.debug('sampling stopping trajectory at stop line')
        # reset cost statistic
        self._min_cost = 10 ** 9
        self._max_cost = 0

        trajectories = list()

        self.sampling_handler.set_s_sampling((x_0_lon[0]+stop_point_s)/2, stop_point_s)

        for t in self.sampling_handler.t_sampling.to_range(samp_level):
            # Longitudinal sampling for all possible velocities
            for s in self.sampling_handler.s_sampling.to_range(samp_level):
                end_state_lon = np.array([s, 0.0, 0.0])
                trajectory_long = QuinticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lon), x_d=end_state_lon)

                # Sample lateral end states (add x_0_lat to sampled states)
                if trajectory_long.coeffs is not None:
                    for d in self.sampling_handler.d_sampling.to_range(samp_level).union({x_0_lat[0]}):
                        end_state_lat = np.array([d, 0.0, 0.0])
                        # SWITCHING TO POSITION DOMAIN FOR LATERAL TRAJECTORY PLANNING
                        if self._LOW_VEL_MODE:
                            s_lon_goal = trajectory_long.evaluate_state_at_tau(t)[0] - x_0_lon[0]
                            if s_lon_goal <= 0:
                                s_lon_goal = t
                            trajectory_lat = QuinticTrajectory(tau_0=0, delta_tau=s_lon_goal, x_0=np.array(x_0_lat),
                                                               x_d=end_state_lat)

                        # Switch to sampling over t for high velocities
                        else:
                            trajectory_lat = QuinticTrajectory(tau_0=0, delta_tau=t, x_0=np.array(x_0_lat),
                                                               x_d=end_state_lat)
                        if trajectory_lat.coeffs is not None:
                            trajectory_sample = TrajectorySample(self.horizon, self.dT, trajectory_long, trajectory_lat,
                                                                 len(trajectories), costMap=self.cost_function.cost_weights)
                            trajectories.append(trajectory_sample)

        # perform pre-check and order trajectories according their cost
        trajectory_bundle = TrajectoryBundle(trajectories, cost_function=cost_function,
                                             multiproc=self._multiproc, num_workers=self._num_workers)
        self._total_count = len(trajectory_bundle._trajectory_bundle)
        msg_logger.debug('%s trajectories sampled' % len(trajectory_bundle._trajectory_bundle))

        return trajectory_bundle

    def convert_state_list_to_commonroad_object(self, state_list: List[ReactivePlannerState], obstacle_id: int = 42):
        """
        Converts a CR trajectory to a CR dynamic obstacle with given dimensions
        :param state_list: trajectory state list of reactive planner
        :param obstacle_id: [optional] ID of ego vehicle dynamic obstacle
        :return: CR dynamic obstacle representing the ego vehicle
        """
        # shift trajectory positions to center
        new_state_list = list()
        for state in state_list:
            new_state_list.append(state.shift_positions_to_center(self.vehicle_params.wb_rear_axle))

        trajectory = Trajectory(initial_time_step=new_state_list[0].time_step, state_list=new_state_list)
        # get shape of vehicle
        shape = Rectangle(self.vehicle_params.length, self.vehicle_params.width)
        # get trajectory prediction
        prediction = TrajectoryPrediction(trajectory, shape)
        return DynamicObstacle(obstacle_id, ObstacleType.CAR, shape, trajectory.state_list[0], prediction)

    def create_coll_object(self, trajectory, vehicle_params, ego_state):
        """Create a collision_object of the trajectory for collision checking with road
        boundary and with other vehicles."""

        collision_object_raw = hf.create_tvobstacle_trajectory(
            traj_list=trajectory,
            box_length=vehicle_params.length / 2,
            box_width=vehicle_params.width / 2,
            start_time_step=ego_state.time_step,
        )
        # if the preprocessing fails, use the raw trajectory
        collision_object, err = trajectory_preprocess_obb_sum(
            collision_object_raw
        )
        if err:
            collision_object = collision_object_raw

        return collision_object

    def check_goal_reached(self):
        # Get the ego vehicle
        self.goal_checker.register_current_state(self.x_0)
        self.goal_status, self.goal_message, self.full_goal_status = self.goal_checker.goal_reached_status()

    def check_collision(self, ego_vehicle):

        ego = pycrcc.TimeVariantCollisionObject((self.x_0.time_step))
        ego.append_obstacle(pycrcc.RectOBB(0.5 * self.vehicle_params.length, 0.5 * self.vehicle_params.width,
                                           ego_vehicle.initial_state.orientation,
                                           ego_vehicle.initial_state.position[0],
                                           ego_vehicle.initial_state.position[1]))

        if not self.collision_checker.collide(ego):
            return False
        else:
            try:
                goal_position = []

                if self.goal_checker.goal.state_list[0].has_value("position"):
                    for x in self.reference_path:
                        if self.goal_checker.goal.state_list[0].position.contains_point(x):
                            goal_position.append(x)
                    s_goal_1, d_goal_1 = self._co.convert_to_curvilinear_coords(goal_position[0][0],
                                                                                goal_position[0][1])
                    s_goal_2, d_goal_2 = self._co.convert_to_curvilinear_coords(goal_position[-2][0],
                                                                                goal_position[-2][1])
                    s_goal = min(s_goal_1, s_goal_2)
                    s_start, d_start = self._co.convert_to_curvilinear_coords(
                        self.planning_problem.initial_state.position[0],
                        self.planning_problem.initial_state.position[1])
                    s_current, d_current = self._co.convert_to_curvilinear_coords(self.x_0.position[0],
                                                                                  self.x_0.position[1])
                    progress = ((s_current - s_start) / (s_goal - s_start))
                elif "time_step" in self.goal_checker.goal.state_list[0].attributes:
                    progress = ((self.x_0.time_step - 1) / self.goal_checker.goal.state_list[0].time_step.end)
                else:
                    msg_logger.error('Could not calculate progress')
                    progress = None
            except:
                progress = None
                msg_logger.error('Could not calculate progress')

            collision_obj = self.collision_checker.find_all_colliding_objects(ego)[0]
            if isinstance(collision_obj, pycrcc.TimeVariantCollisionObject):
                obj = collision_obj.obstacle_at_time((self.x_0.time_step - 1))
                center = obj.center()
                last_center = collision_obj.obstacle_at_time(self.x_0.time_step - 2).center()
                r_x = obj.r_x()
                r_y = obj.r_y()
                orientation = obj.orientation()
                self.logger.log_collision(True, self.vehicle_params.length, self.vehicle_params.width, progress, center,
                                          last_center, r_x, r_y, orientation)
            else:
                self.logger.log_collision(False, self.vehicle_params.length, self.vehicle_params.width, progress)
            return True

    def shift_orientation(self, trajectory: Trajectory, interval_start=-np.pi, interval_end=np.pi):
        for state in trajectory.state_list:
            while state.orientation < interval_start:
                state.orientation += 2 * np.pi
            while state.orientation > interval_end:
                state.orientation -= 2 * np.pi
        return trajectory

