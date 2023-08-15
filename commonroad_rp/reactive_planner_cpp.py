__author__ = "Rainer Trauth, Gerald Würsching"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# python packages
import math
import time

import numpy as np
import copy
from typing import List, Optional, Tuple
import logging
from risk_assessment.risk_costs import calc_risk

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
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectory_preprocess_obb_sum, trajectories_collision_static_obstacles

# commonroad_rp imports
from commonroad_rp.sampling_matrix import SamplingHandler, generate_sampling_matrix

from commonroad_rp.utility.utils_coordinate_system import CoordinateSystem, smooth_ref_path

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
        self.reference_path = None
        self.predictionsForCpp = {}
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

        # *****************************
        # C++ Trajectory Handler Import
        # *****************************

        self.handler: TrajectoryHandler = TrajectoryHandler(dt=config.planning.dt)
        self.cost_weights = OmegaConf.to_object(config.cost.cost_weights)
        self.coordinate_system: CoordinateSystemWrapper = CoordinateSystemWrapper
        self.trajectory_handler_set_constant_functions()
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

    @property
    def goal_checker(self):
        """Return the goal checker."""
        return self._goal_checker

    def _check_valid_settings(self):
        """Checks validity of provided dt and horizon"""
        assert is_positive(self.dT), 'provided dt is not correct! dt = {}'.format(self.dT)
        assert is_positive(self.N) and is_natural_number(self.N), 'N is not correct!'
        assert is_positive(self.horizon), 'provided t_h is not correct! dt = {}'.format(self.horizon)

    @property
    def collision_checker(self) -> pycrcc.CollisionChecker:
        return self._cc

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
                         cost_weights=None, occlusion_module=None, desired_velocity: float = None,
                         predictions=None, reach_set=None, behavior=None):
        """
        Sets all external information in reactive planner
        :param scenario: Commonroad scenario
        :param reference_path: reference path as polyline
        :param planning_problem: reference path as polyline
        :param goal_area: commonroad goal area
        :param x_0: current ego vehicle state in global coordinate system
        :param x_cl: current ego vehicle state in curvilinear coordinate system
        :param cost_weights: current used cost weights
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
            # self.set_x_cl(x_cl)
        if cost_weights is not None:
            self.set_cost_function(cost_weights)
        if occlusion_module is not None:
            self.set_occlusion_module(occlusion_module)
        if desired_velocity is not None:
            self.set_desired_velocity(desired_velocity, x_0.velocity)
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
        for key in self.predictions.keys():
            predictedPath = []

            for time_step in range(self.predictions[key]['pos_list'].shape[0]):
                position = np.append(self.predictions[key]['pos_list'][time_step], 0)
                orientation = np.zeros(shape=(4))
                orientation[2] = np.sin(self.predictions[key]['orientation_list'][time_step] / 2.0)
                orientation[3] = np.cos(self.predictions[key]['orientation_list'][time_step] / 2.0)
                covariance = np.zeros(shape=(6, 6))
                covariance[:2, :2] = self.predictions[key]['cov_list'][time_step]

                pwc = PoseWithCovariance(position, orientation, covariance)
                predictedPath.append(pwc)

            self.predictionsForCpp[key] = PredictedObject(key, predictedPath)

    def set_reach_set(self, reach_set):
        self.reach_set = reach_set

    def set_x_0(self, x_0: ReactivePlannerState):
        # set Cartesian initial state
        self.x_0 = x_0
        if self.x_0.velocity < self._low_vel_mode_threshold:
            self._LOW_VEL_MODE = True
        else:
            self._LOW_VEL_MODE = False

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

    def set_cost_function(self, cost_weights):
        self.config.cost.cost_weights = cost_weights
        self.trajectory_handler_set_constant_functions()
        self.trajectory_handler_set_changing_functions()
        self.logger.set_logging_header(self.config.cost.cost_weights)

    def trajectory_handler_set_constant_functions(self):
        self.handler.add_feasability_function(CheckYawRateConstraint(deltaMax=self.vehicle_params.delta_max,
                                                                     wheelbase=self.vehicle_params.wheelbase, wholeTrajectory=False))
        self.handler.add_feasability_function(CheckAccelerationConstraint(switchingVelocity=self.vehicle_params.v_switch,
                                                                          maxAcceleration=self.vehicle_params.a_max, wholeTrajectory=False))
        self.handler.add_feasability_function(CheckCurvatureConstraint(deltaMax=self.vehicle_params.delta_max,
                                                                       wheelbase=self.vehicle_params.wheelbase, wholeTrajectory=False))
        self.handler.add_feasability_function(CheckCurvatureRateConstraint(wheelbase=self.vehicle_params.wheelbase,
                                                                           velocityDeltaMax=self.vehicle_params.v_delta_max, wholeTrajectory=False))

        name = "acceleration"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateAccelerationCost(name, self.cost_weights[name]))

        name = "jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateJerkCost(name, self.cost_weights[name]))

        name = "lateral_jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateLateralJerkCost(name, self.cost_weights[name]))

        name = "longitudinal_jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateLongitudinalJerkCost(name, self.cost_weights[name]))

        name = "orientation_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateOrientationOffsetCost(name, self.cost_weights[name]))

        name = "lane_center_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateLaneCenterOffsetCost(name, self.cost_weights[name]))

        name = "distance_to_reference_path"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateDistanceToReferencePathCost(name, self.cost_weights[name]))

    def trajectory_handler_set_changing_functions(self):

        self.handler.add_function(FillCoordinates(lowVelocityMode=self._LOW_VEL_MODE,
                                                  initialOrientation=self.x_0.orientation,
                                                  coordinateSystem=self.coordinate_system,
                                                  horizon=int(self.config.planning.planning_horizon)))
        name = "prediction"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateCollisionProbabilityMahalanobis(name, self.cost_weights[name], self.predictionsForCpp))

        name = "distance_to_obstacles"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            obstacle_positions = np.zeros((len(self.scenario.obstacles), 2))
            for i, obstacle in enumerate(self.scenario.obstacles):
                state = obstacle.state_at_time(self.x_0.time_step)
                if state is not None:
                    obstacle_positions[i, 0] = state.position[0]
                    obstacle_positions[i, 1] = state.position[1]

            self.handler.add_cost_function(CalculateDistanceToObstacleCost(name, self.cost_weights[name], obstacle_positions))

        name = "velocity_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(CalculateVelocityOffsetCost(name, self.cost_weights[name], self._desired_speed))

    def set_reference_path(self, reference_path: np.ndarray):
        """
        Automatically creates a curvilinear coordinate system from a given reference path
        :param reference_path: reference_path as polyline
        """

        self.reference_path = smooth_ref_path(reference_path)
        self.coordinate_system: CoordinateSystemWrapper = CoordinateSystemWrapper(copy.deepcopy(self.reference_path))
        self._co: CoordinateSystem = CoordinateSystem(self.reference_path)
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

    def set_desired_velocity(self, desired_velocity: float, current_speed: float = None, stopping: bool = False,
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
        self._infeasible_count_kinematics = np.zeros(10)
        self._infeasible_count_collision = 0
        self.infeasible_kinematics_percentage = 0
        # **************************************
        # Initialization of Cpp Frenet Functions
        # **************************************
        self.trajectory_handler_set_changing_functions()
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

        msg_logger.debug(f'Initial x_0 lon = {x_0_lon}')
        msg_logger.debug(f'Initial x_0 lat = {x_0_lat}')

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
            t1_range = np.array(list(self.sampling_handler.t_sampling.to_range(samp_level)))
            ss1_range = np.array(list(self.sampling_handler.v_sampling.to_range(samp_level)))
            d1_range = np.array(list(self.sampling_handler.d_sampling.to_range(samp_level).union(x_0_lat[0])))

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

            if not self.config.debug.multiproc or self.config.multiagent.multiprocessing:
                self.handler.evaluate_all_current_functions(True)
            else:
                self.handler.evaluate_all_current_functions_concurrent(True)

            feasible_trajectories = []
            infeasible_trajectories = []
            for trajectory in self.handler.get_sorted_trajectories():
                # check if trajectory is feasible
                if trajectory.feasible:
                    feasible_trajectories.append(trajectory)
                elif trajectory.valid:
                    infeasible_trajectories.append(trajectory)

            self.infeasible_kinematics_percentage = float(len(feasible_trajectories)
                                                    / (len(feasible_trajectories) + len(infeasible_trajectories))) * 100
            # print size of feasible trajectories and infeasible trajectories
            msg_logger.info('<Reactive Planner>: Found {} feasible trajectories and {} infeasible trajectories'.format(feasible_trajectories.__len__(), infeasible_trajectories.__len__()))
            msg_logger.debug(
                'Percentage of valid & feasible trajectories: %s %%' % str(self.infeasible_kinematics_percentage))
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

            self.transfer_infeasible_logging_information(infeasible_trajectories)

            msg_logger.debug('Rejected {} infeasible trajectories due to kinematics'.format(
                self.infeasible_count_kinematics))
            msg_logger.debug('Rejected {} infeasible trajectories due to collisions'.format(
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
                    s_goal_1, d_goal_1 = self.coordinate_system.system.convert_to_curvilinear_coords(goal_position[0][0],
                                                                                                     goal_position[0][1])
                    s_goal_2, d_goal_2 = self.coordinate_system.system.convert_to_curvilinear_coords(goal_position[-2][0],
                                                                                                     goal_position[-2][1])
                    s_goal = min(s_goal_1, s_goal_2)
                    s_start, d_start = self.coordinate_system.system.convert_to_curvilinear_coords(
                        self.planning_problem.initial_state.position[0],
                        self.planning_problem.initial_state.position[1])
                    s_current, d_current = self.coordinate_system.system.convert_to_curvilinear_coords(self.x_0.position[0],
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

    def transfer_infeasible_logging_information(self, infeasible_trajectories):

        feas_list = [i.feasabilityMap['Curvature Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self.infeasible_count_kinematics[5] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Yaw rate Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self.infeasible_count_kinematics[6] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Curvature Rate Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self.infeasible_count_kinematics[7] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Acceleration Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self.infeasible_count_kinematics[8] = int(sum(acc_feas))

        self.infeasible_count_kinematics[0] = int(sum(self.infeasible_count_kinematics))
