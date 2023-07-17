__author__ = "Rainer Trauth, Gerald Würsching, Christian Pek"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "0.5"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# python packages
import time

import numpy as np
from typing import List, Optional, Tuple
import logging
from risk_assessment.risk_costs import calc_risk

import copy
# commonroad-io
from commonroad.common.validity import *
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InputState
from commonroad.scenario.scenario import Scenario

# commonroad_dc
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectory_preprocess_obb_sum, trajectories_collision_static_obstacles

# commonroad_rp imports
from commonroad_rp.parameter import TimeSampling, VelocitySampling, PositionSampling

from commonroad_rp.utility.utils_coordinate_system import CoordinateSystem, smooth_ref_path
from commonroad_rp.utility import reachable_set
from commonroad_rp.state import ReactivePlannerState

from cr_scenario_handler.utils.goalcheck import GoalReachedChecker
from cr_scenario_handler.utils.configuration import Configuration
from commonroad_rp.utility.logging_helpers import DataLoggingCosts

from commonroad_rp.prediction_helpers import collision_checker_prediction
from commonroad_rp.utility import helper_functions as hf
from risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle

from commonroad_rp.utility.load_json import (
    load_harm_parameter_json,
    load_risk_json
)

from commonroad_rp.sampling_matrix import generate_sampling_matrix
from omegaconf import OmegaConf

from frenetPlannerHelper.trajectory_functions.feasability_functions import *
from frenetPlannerHelper.trajectory_functions.cost_functions import *
from frenetPlannerHelper.trajectory_functions import FillCoordinates, ComputeInitialState
from frenetPlannerHelper import *

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
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.predictions = None
        self.predictionsForCpp = {}
        self.behavior = None
        self.set_new_ref_path = None
        self.cost_function = None
        self._co: Optional[CoordinateSystem] = None
        self._cc: Optional[pycrcc.CollisionChecker] = None
        self.goal_status = False
        self.full_goal_status = None
        self.goal_area = None
        self.occlusion_module = None
        self.goal_message = "Planner is in time step 0!"
        self.use_amazing_visualizer = config.debug.use_amazing_visualizer

        self._desired_speed = None
        self._desired_d = 0.
        self.max_seen_costs = 1

        # **************************
        # Handler import
        # **************************

        self.handler: TrajectoryHandler = TrajectoryHandler(dt=config.planning.dt)
        self.params = OmegaConf.to_object(config.cost.params)
        self.coordinate_system: CoordinateSystemWrapper
        self.set_handler_constant_functions()
        # **************************
        # Extensions Initialization
        # **************************
        if config.prediction.mode:
            self.use_prediction = True
        else:
            self.use_prediction = False

        self.set_collision_checker(self.scenario)
        self._goal_checker = GoalReachedChecker(planning_problem)

        # **************************
        # Statistics Initialization
        # **************************
        self._total_count = 0
        self._infeasible_count_collision = 0
        self._infeasible_count_kinematics = np.zeros(10)
        self._optimal_cost = 0

        # **************************
        # Sampling Initialization
        # **************************
        # Set Sampling Parameters#
        self._sampling_min = config.sampling.sampling_min
        self._sampling_max = config.sampling.sampling_max
        self._sampling_d = None
        self._sampling_t = None
        self._sampling_v = None
        self.set_d_sampling_parameters(config.sampling.d_min, config.sampling.d_max)
        self.set_t_sampling_parameters(config.sampling.t_min, config.planning.dt,
                                       config.planning.planning_horizon)

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
                                       cost_params=config.cost.params.cluster0)
        self._draw_traj_set = config.debug.draw_traj_set
        self._kinematic_debug = config.debug.kinematic_debug

        # **************************
        # Risk & Harm Initialization
        # **************************
        self.params_harm = load_harm_parameter_json(work_dir)
        self.params_risk = load_risk_json(work_dir)

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

        # check whether reachable sets have to be calculated for responsibility
        if (
                'R' in config.cost.params.cluster0
                and config.cost.params.cluster0['R'] > 0
        ):
            self.responsibility = True
            self.reach_set = reachable_set.ReachSet(
                scenario=self.scenario,
                ego_id=24,
                ego_length=config.vehicle.length,
                ego_width=config.vehicle.width,
                work_dir=work_dir
            )
        else:
            self.responsibility = False
            self.reach_set = None

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

    def update_externals(self, x_0: ReactivePlannerState=None, x_cl: Optional[Tuple[List, List]]=None,
                         scenario: Scenario=None, goal_area=None, planning_problem=None,
                         cost_function=None, reference_path: np.ndarray=None, occlusion_module=None,
                         desired_velocity: float = None, predictions=None, behavior=None):
        if x_0 is not None:
            self.x_0 = x_0

            # Check for low velocity mode
            if self.x_0.velocity < self._low_vel_mode_threshold:
                self._LOW_VEL_MODE = True
            else:
                self._LOW_VEL_MODE = False

        if x_cl is not None:
            self.x_cl = x_cl
        if scenario is not None:
            self.set_scenario(scenario)
        if goal_area is not None:
            self.set_goal_area(goal_area)
        if planning_problem is not None:
            self.set_planning_problem(planning_problem)
        if cost_function is not None:
            self.set_cost_function(cost_function)
        if occlusion_module is not None:
            self.set_occlusion_module(occlusion_module)
        if reference_path is not None:
            self.set_reference_path(reference_path)
        if desired_velocity is not None:
            self.set_desired_velocity(desired_velocity, x_0.velocity)
        if predictions is not None:
            self.predictions = predictions
            for key in self.predictions.keys():
                self.predictionsForCpp[key] = PredictedObject(self.predictions[key]['pos_list'].shape[0])
                for time_step in range(self.predictions[key]['pos_list'].shape[0]):
                    self.predictionsForCpp[key].object_id = key
                    self.predictionsForCpp[key].predictedPath[time_step].position = np.append(self.predictions[key]['pos_list'][time_step], 0)
                    self.predictionsForCpp[key].predictedPath[time_step].orientation[2] = np.sin(self.predictions[key]['orientation_list'][time_step] / 2.0)
                    self.predictionsForCpp[key].predictedPath[time_step].orientation[3] = np.cos(self.predictions[key]['orientation_list'][time_step] / 2.0)
                    self.predictionsForCpp[key].predictedPath[time_step].covariance[:2, :2] = self.predictions[key]['cov_list'][time_step]


        if behavior is not None:
            self.behavior = behavior

    def set_scenario(self, scenario: Scenario):
        """Update the scenario to synchronize between agents"""
        self.scenario = scenario
        self.set_collision_checker(scenario)

    def set_predictions(self, predictions: dict):
        self.predictions = predictions

    def set_x_0(self, x_0: ReactivePlannerState):
        # set Cartesian initial state
        self.x_0 = x_0

    def set_x_cl(self, x_cl):
        # set Curvlinear initial state
        self.x_cl = x_cl

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

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function
        self.logger.set_logging_header(cost_function.PartialCostFunctionMapping)

    def set_handler_constant_functions(self):
        self.handler.add_feasability_function(CheckYawRateConstraint(deltaMax=self.vehicle_params.delta_max, wheelbase=self.vehicle_params.wheelbase))
        self.handler.add_feasability_function(CheckAccelerationConstraint(switchingVelocity=self.vehicle_params.v_switch, maxAcceleration=self.vehicle_params.a_max))
        self.handler.add_feasability_function(CheckCurvatureConstraint(deltaMax=self.vehicle_params.delta_max, wheelbase=self.vehicle_params.wheelbase))
        self.handler.add_feasability_function(CheckCurvatureRateConstraint(wheelbase=self.vehicle_params.wheelbase, velocityDeltaMax=self.vehicle_params.v_delta_max))

        name = "Acceleration"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateAccelerationCost(name, self.params["cluster0"][name]))

        name = "Jerk"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateJerkCost(name, self.params["cluster0"][name]))

        name = "Lateral Jerk"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateLateralJerkCost(name, self.params["cluster0"][name]))

        name = "Longitudinal Jerk"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateLongitudinalJerkCost(name, self.params["cluster0"][name]))

        name = "Orientation Offset"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateOrientationOffsetCost(name, self.params["cluster0"][name]))

        name = "Path Length"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            msg_logger.info("Path Length not implemented yet")

        name = "Lane Center Offset"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateLaneCenterOffsetCost(name, self.params["cluster0"][name]))

        name = "Velocity Costs"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            msg_logger.info("Velocity Costs not implemented yet")

        name = "Distance to Reference Path"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateDistanceToReferencePathCost(name, self.params["cluster0"][name]))

    def set_handler_changing_functions(self):

        self.handler.add_function(FillCoordinates(lowVelocityMode=self._LOW_VEL_MODE,
                                                  initialOrientation=self.x_0.orientation,
                                                  coordinateSystem=self.coordinate_system))
        name = "Prediction"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateCollisionProbabilityMahalanobis(name, self.params["cluster0"][name], self.predictionsForCpp))

        name = "Distance to Obstacles"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            obstacle_positions = np.zeros((len(self.scenario.obstacles), 2))
            for i, obstacle in enumerate(self.scenario.obstacles):
                state = obstacle.state_at_time(self.x_0.time_step)
                if state is not None:
                    obstacle_positions[i, 0] = state.position[0]
                    obstacle_positions[i, 1] = state.position[1]

            self.handler.add_cost_function(CalculateDistanceToObstacleCost(name, self.params["cluster0"][name], obstacle_positions))

        name = "Velocity Offset"
        if name in self.params["cluster0"].keys() and self.params["cluster0"][name] > 0:
            self.handler.add_cost_function(CalculateVelocityOffsetCost(name, self.params["cluster0"][name], self._desired_speed))

    def set_reference_path(self, reference_path: np.ndarray):
        """
        Automatically creates a curvilinear coordinate system from a given reference path
        :param reference_path: reference_path as polyline
        """

        reference_path = smooth_ref_path(reference_path)
        self.coordinate_system: CoordinateSystemWrapper = CoordinateSystemWrapper(reference_path)
        self._co: CoordinateSystem = CoordinateSystem(reference_path)

    def set_goal_area(self, goal_area):
        """
        Sets the planning problem
        :param planning_problem: PlanningProblem
        """
        self.goal_area = goal_area

    def set_occlusion_module(self, occ_module):
        self.occlusion_module = occ_module

    def set_planning_problem(self, planning_problem):
        """
        Sets the planning problem
        :param planning_problem: PlanningProblem
        """
        self.planning_problem = planning_problem

    def set_t_sampling_parameters(self, t_min, dt, horizon):
        """
        Sets sample parameters of time horizon
        :param t_min: minimum of sampled time horizon
        :param dt: length of each sampled step
        :param horizon: sampled time horizon
        """
        self._sampling_t = TimeSampling(t_min, horizon, self._sampling_max, dt)
        self.N = int(round(horizon / dt))
        self.horizon = horizon

    def set_d_sampling_parameters(self, delta_d_min, delta_d_max):
        """
        Sets sample parameters of lateral offset
        :param delta_d_min: lateral distance lower than reference
        :param delta_d_max: lateral distance higher than reference
        """
        self._sampling_d = PositionSampling(delta_d_min, delta_d_max, self._sampling_max)

    def set_v_sampling_parameters(self, v_min, v_max):
        """
        Sets sample parameters of sampled velocity interval
        :param v_min: minimal velocity sample bound
        :param v_max: maximal velocity sample bound
        """
        self._sampling_v = VelocitySampling(v_min, v_max, self._sampling_max)

    def set_desired_velocity(self, desired_velocity: float, current_speed: float = None, stopping: bool = False,
                             v_limit: float = 80):
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

        self._sampling_v = VelocitySampling(min_v, max_v, self._sampling_max)

        msg_logger.info('Sampled interval of velocity: {} m/s - {} m/s'.format(min_v, max_v))

    def _get_no_of_samples(self, samp_level: int) -> int:
        """
        Returns the number of samples for a given sampling level
        :param samp_level: The sampling level
        :return: Number of trajectory samples for given sampling level
        """
        return len(self._sampling_v.to_range(samp_level)) * len(self._sampling_d.to_range(samp_level)) * len(
            self._sampling_t.to_range(samp_level))

    def _create_coll_object(self, trajectory, vehicle_params, ego_state):
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

    # def _create_stopping_trajectory(self, x_0, x_0_lon, x_0_lat, stop_point, cost_function):
    #
    #     return trajectory_bundle

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

    def plan(self) -> tuple:
        """
        Plans an optimal trajectory
        :return: Optimal trajectory as tuple
        """

        # # Assign responsibility to predictions
        # if self.responsibility:
        #     self.predictions = assign_responsibility_by_action_space(
        #         self.scenario, self.x_0, self.predictions
        #     )
        #     # calculate reachable sets
        #     self.reach_set.calc_reach_sets(self.x_0, list(self.predictions.keys()))

        # **************************************
        # Initialization of Cpp Frenet Functions
        # **************************************
        self.set_handler_changing_functions()
        initial_state = TrajectorySample(x0=self.x_0.position[0],
                                         y0=self.x_0.position[1],
                                         orientation0=self.x_0.orientation,
                                         acceleration0=self.x_0.acceleration,
                                         velocity0=self.x_0.velocity)

        initial_state_computation = ComputeInitialState(coordinateSystem=self.coordinate_system,
                                                        wheelBase=self.vehicle_params.wheelbase,
                                                        steeringAngle=self.x_0.steering_angle,
                                                        lowVelocityMode=self._LOW_VEL_MODE)

        initial_state_computation.evaluate_trajectory(initial_state)

        x_0_lat = [initial_state.curvilinear.d, initial_state.curvilinear.d_dot, initial_state.curvilinear.d_ddot]
        x_0_lon = [initial_state.curvilinear.s, initial_state.curvilinear.s_dot, initial_state.curvilinear.s_ddot]

        msg_logger.debug("<ReactivePlanner>: Starting planning with: \n#################")
        msg_logger.debug(f'Initial x_0 lon = {x_0_lon}')
        msg_logger.debug(f'Initial x_0 lat = {x_0_lat}')
        msg_logger.debug("#################")

        msg_logger.debug('<Reactive Planner>: initial state is: lon = {} / lat = {}'.format(x_0_lon, x_0_lat))
        msg_logger.debug('<Reactive Planner>: desired velocity is {} m/s'.format(self._desired_speed))

        # Initialization of while loop
        optimal_trajectory = None
        feasible_trajectories = []
        t0 = time.time()
        samp_level = self._sampling_min
        while optimal_trajectory is None and samp_level < self._sampling_max:

            # *************************************
            # Create & Evaluate Trajectories in Cpp
            # *************************************
            t1_range = np.array(list(self._sampling_t.to_range(samp_level)))
            ss1_range = np.array(list(self._sampling_v.to_range(samp_level)))
            d1_range = np.array(list(self._sampling_d.to_range(samp_level).union(x_0_lat[0])))

            sampling_matrix = generate_sampling_matrix(t0_range=0.0,
                                                       t1_range=t1_range,
                                                       s0_range=x_0_lon[0],
                                                       ss0_range=x_0_lon[1],
                                                       sss0_range=x_0_lon[2],
                                                       ss1_range=ss1_range,
                                                       sss1_range=0,
                                                       d0_range=x_0_lat[0],
                                                       dd0_range=x_0_lat[1],
                                                       ddd0_range=x_0_lat[2],
                                                       d1_range=d1_range,
                                                       dd1_range=0.0,
                                                       ddd1_range=0.0)

            self.handler.reset_Trajectories()
            self.handler.generate_trajectories(sampling_matrix, self._LOW_VEL_MODE)
            self.handler.evaluate_all_current_functions_concurrent(True)
            #self.handler.evaluate_all_current_functions(True)

            feasible_trajectories = []
            infeasible_trajectories = []
            for trajectory in self.handler.get_sorted_trajectories():
                # check if trajectory is feasible
                if trajectory.feasible:
                    feasible_trajectories.append(trajectory)
                else:
                    infeasible_trajectories.append(trajectory)

            # for visualization store all trajectories with validity level based on kinematic validity
            if self._draw_traj_set or self.save_all_traj or self.use_amazing_visualizer:
                trajectories = feasible_trajectories + infeasible_trajectories
                self.all_traj = trajectories

            # *****************************
            # Optional: Use Occlusion Model
            # *****************************
            if self.use_occ_model and feasible_trajectories:
                self.occlusion_module.occ_phantom_module.evaluate_trajectories(feasible_trajectories)
                self.occlusion_module.occ_uncertainty_map_evaluator.evaluate_trajectories(feasible_trajectories)

            # ******************************************
            # Check Feasible Trajectories for Collisions
            # ******************************************
            optimal_trajectory = self.trajectory_collision_check(feasible_trajectories)

            if optimal_trajectory is not None and self.log_risk:
                optimal_trajectory = self.set_risk_costs(optimal_trajectory)

            msg_logger.debug('<ReactivePlanner>: Rejected {} infeasible trajectories due to kinematics'.format(
                self.infeasible_count_kinematics))
            msg_logger.debug('<ReactivePlanner>: Rejected {} infeasible trajectories due to collisions'.format(
                self.infeasible_count_collision))

            # increase sampling level (i.e., density) if no optimal trajectory could be found
            samp_level += 1

        planning_time = time.time() - t0
        self.optimal_trajectory = optimal_trajectory

        # **************************
        # Set Risk Costs
        # **************************
        if optimal_trajectory is None and feasible_trajectories:
            for traje in feasible_trajectories:
                self.set_risk_costs(traje)
            sort_risk = sorted(feasible_trajectories, key=lambda traj: traj._ego_risk + traj._obst_risk, reverse=False)
            optimal_trajectory = sort_risk[0]

        # ******************************************
        # Update Trajectory Pair & Commonroad Object
        # ******************************************
        trajectory_pair = self._compute_trajectory_pair(optimal_trajectory) if optimal_trajectory is not None else None
        if trajectory_pair is not None:
            current_ego_vehicle = self.convert_state_list_to_commonroad_object(trajectory_pair[0].state_list)
            self.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

        # **************************
        # Logging
        # **************************
        if optimal_trajectory is not None:
            self.logger.trajectory_number = self.x_0.time_step
            self.logger.log(optimal_trajectory, infeasible_kinematics=self.infeasible_count_kinematics,
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

    def trajectory_collision_check(self, feasible_trajectories):
        """
        Checks valid trajectories for collisions with static obstacles
        :param feasible_trajectories: feasible trajectories list
        :return trajectory: optimal feasible trajectory or None
        """
        # go through sorted list of sorted trajectories and check for collisions
        for trajectory in feasible_trajectories:
            # Add Occupancy of Trajectory to do Collision Checks later
            cart_traj = self._compute_cart_traj(trajectory)
            occupancy = self.convert_state_list_to_commonroad_object(cart_traj.state_list)
            # get collision_object
            coll_obj = self._create_coll_object(occupancy, self.vehicle_params, self.x_0)

            # TODO: Check kinematic checks in cpp. no valid traj available
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

    def convert_state_list_to_commonroad_object(self, state_list, obstacle_id: int = 42):
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

    def check_goal_reached(self):
        # Get the ego vehicle
        self.goal_checker.register_current_state(self.x_0)
        self.goal_status, self.goal_message, self.full_goal_status = self.goal_checker.goal_reached_status()

    def check_collision(self, ego_vehicle):

        ego = pycrcc.TimeVariantCollisionObject((self.x_0.time_step))
        ego.append_obstacle(pycrcc.RectOBB(0.5 * self.vehicle_params.length, 0.5 * self.vehicle_params.width,
                                           ego_vehicle.initial_state.orientation,
                                           ego_vehicle.initial_state.position[0], ego_vehicle.initial_state.position[1]))

        if not self.collision_checker.collide(ego):
            return False
        else:
            try:
                goal_position = []

                if self.goal_checker.goal.state_list[0].has_value("position"):
                    for x in self.reference_path:
                        if self.goal_checker.goal.state_list[0].position.contains_point(x):
                            goal_position.append(x)
                    s_goal_1, d_goal_1 = self._co.convert_to_curvilinear_coords(goal_position[0][0], goal_position[0][1])
                    s_goal_2, d_goal_2 = self._co.convert_to_curvilinear_coords(goal_position[-1][0], goal_position[-1][1])
                    s_goal = min(s_goal_1, s_goal_2)
                    s_start, d_start = self._co.convert_to_curvilinear_coords(
                        self.planning_problem.initial_state.position[0],
                        self.planning_problem.initial_state.position[1])
                    s_current, d_current = self._co.convert_to_curvilinear_coords(self.x_0.position[0], self.x_0.position[1])
                    progress = ((s_current - s_start) / (s_goal - s_start))
                elif "time_step" in self.goal_checker.goal.state_list[0].attributes:
                    progress = ((self.x_0.time_step -1) / self.goal_checker.goal.state_list[0].time_step.end)
                else:
                    msg_logger.error('Could not calculate progress')
                    progress = None
            except:
                progress = None
                msg_logger.error('Could not calculate progress')

            collision_obj = self.collision_checker.find_all_colliding_objects(ego)[0]
            if isinstance(collision_obj, pycrcc.TimeVariantCollisionObject):
                obj = collision_obj.obstacle_at_time((self.x_0.time_step -1))
                center = obj.center()
                last_center = collision_obj.obstacle_at_time(self.x_0.time_step-2).center()
                r_x = obj.r_x()
                r_y = obj.r_y()
                orientation = obj.orientation()
                self.logger.log_collision(True, self.vehicle_params.length, self.vehicle_params.width, progress, center,
                                          last_center, r_x, r_y, orientation)
            else:
                self.logger.log_collision(False, self.vehicle_params.length, self.vehicle_params.width, progress)
            return True

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

    def shift_orientation(self, trajectory: Trajectory, interval_start=-np.pi, interval_end=np.pi):
        for state in trajectory.state_list:
            while state.orientation < interval_start:
                state.orientation += 2 * np.pi
            while state.orientation > interval_end:
                state.orientation -= 2 * np.pi
        return trajectory
