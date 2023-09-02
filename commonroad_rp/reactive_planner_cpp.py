__author__ = "Rainer Trauth, Gerald Würsching"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# python packages
import time
import numpy as np
import copy
import logging

# commonroad_dc
import commonroad_dc.pycrcc as pycrcc

# commonroad_rp imports
from commonroad_rp.sampling_matrix import generate_sampling_matrix
from commonroad_rp.utility.utils_coordinate_system import CoordinateSystem, smooth_ref_path

from frenetPlannerHelper.trajectory_functions.feasability_functions import *
from frenetPlannerHelper.trajectory_functions.cost_functions import *
from frenetPlannerHelper.trajectory_functions import FillCoordinates, ComputeInitialState
from frenetPlannerHelper import *

from commonroad_rp.planner import Planner

# get logger
msg_logger = logging.getLogger("Message_logger")


class ReactivePlannerCpp(Planner):
    """
    Reactive planner class that plans trajectories in a sampling-based fashion
    """
    def __init__(self, config, scenario, planning_problem, log_path, work_dir):
        """
        Constructor of the reactive planner
        : param config: Configuration object holding all planner-relevant configurations
        """
        super().__init__(config, scenario, planning_problem, log_path, work_dir)

        self.predictionsForCpp = {}

        # *****************************
        # C++ Trajectory Handler Import
        # *****************************

        self.handler: TrajectoryHandler = TrajectoryHandler(dt=self.config.planning.dt)
        self.coordinate_system: CoordinateSystemWrapper = CoordinateSystemWrapper
        self.trajectory_handler_set_constant_functions()

        # **************************
        # Statistics Initialization
        # **************************
        self._infeasible_count_kinematics = np.zeros(10)

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
        self.logger.sql_logger.write_reference_path(reference_path)

    def plan(self) -> tuple:
        """
        Plans an optimal trajectory
        :return: Optimal trajectory as tuple
        """
        self._infeasible_count_kinematics = np.zeros(10)
        self._collision_counter = 0
        self.infeasible_kinematics_percentage = 0
        # **************************************
        # Initialization of Cpp Frenet Functions
        # **************************************
        self.trajectory_handler_set_changing_functions()
        x_0_lon = None
        x_0_lat = None
        if self.x_cl is None:
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

        else:
            x_0_lon = self.x_cl[0]
            x_0_lat = self.x_cl[1]


        msg_logger.debug(f'Initial x_0 lon = {x_0_lon}')
        msg_logger.debug(f'Initial x_0 lat = {x_0_lat}')

        msg_logger.debug('Initial state is: lon = {} / lat = {}'.format(x_0_lon, x_0_lat))
        msg_logger.debug('Desired velocity is {} m/s'.format(self._desired_speed))

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
            d1_range = np.array(list(self.sampling_handler.d_sampling.to_range(samp_level).union({x_0_lat[0]})))

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

            if not self.config.debug.multiproc or (self.config.multiagent.multiprocessing and
                                                   self.config.multiagent.use_multiagent):
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
            msg_logger.debug('Found {} feasible trajectories and {} infeasible trajectories'.format(feasible_trajectories.__len__(), infeasible_trajectories.__len__()))
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
            msg_logger.warning("No optimal trajectory available. Select lowest risk trajectory!")
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
