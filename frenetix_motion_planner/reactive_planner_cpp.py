__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# python packages
import time
import numpy as np
from itertools import product
from typing import List
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt

# frenetix_motion_planner imports
from frenetix_motion_planner.sampling_matrix import generate_sampling_matrix
from frenetix_motion_planner.sampling_matrix import Sampling

from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem
from cr_scenario_handler.utils.visualization import visualize_scenario_and_pp

import frenetix
import frenetix.trajectory_functions
import frenetix.trajectory_functions.feasability_functions as ff
import frenetix.trajectory_functions.cost_functions as cf

from frenetix_motion_planner.planner import Planner
from frenetix_motion_planner.sampling_matrix import Sampling


class ReactivePlannerCpp(Planner):
    """
    Reactive planner class that plans trajectories in a sampling-based fashion
    """
    def __init__(self, config_plan, config_sim, scenario, planning_problem, log_path, work_dir, msg_logger):
        """
        Constructor of the reactive planner
        : param config_plan: Configuration object holding all planner-relevant configurations
        """
        super().__init__(config_plan, config_sim, scenario, planning_problem, log_path, work_dir, msg_logger)

        self.predictionsForCpp = {}

        # *****************************
        # C++ Trajectory Handler Import
        # *****************************

        self.handler: frenetix.TrajectoryHandler = frenetix.TrajectoryHandler(dt=self.config_plan.planning.dt)
        self.coordinate_system_cpp: frenetix.CoordinateSystemWrapper
        self.user_input = self.config_plan.planning.spacing_trajs
        self.count = 0
        self.total_time = 0
        self.trajectory_handler_set_constant_cost_functions()
        self.trajectory_handler_set_constant_feasibility_functions()

    def set_predictions(self, predictions: dict):
        self.use_prediction = True
        self.predictions = predictions
        for key, pred in self.predictions.items():
            num_steps = pred['pos_list'].shape[0]
            predicted_path: List[frenetix.PoseWithCovariance] = [None] * num_steps

            for time_step in range(num_steps):
                # Ensure the position is in float64 format
                position = np.append(pred['pos_list'][time_step].astype(np.float64), [0.0]).astype(np.float64)

                # Preallocate orientation array and fill in the values
                orientation = np.zeros(4, dtype=np.float64)
                orientation[2:] = np.array([np.sin(pred['orientation_list'][time_step] / 2.0),
                                            np.cos(pred['orientation_list'][time_step] / 2.0)], dtype=np.float64)

                # Symmetrize the covariance matrix if necessary and convert to float64
                covariance = pred['cov_list'][time_step].astype(np.float64)
                # if not np.array_equal(covariance, covariance.T):
                # covariance = ((covariance + covariance.T) / 2).astype(np.float64)

                # Create the covariance matrix for PoseWithCovariance
                covariance_matrix = np.zeros((6, 6), dtype=np.float64)
                covariance_matrix[:2, :2] = covariance

                # Create PoseWithCovariance object and add to predicted_path
                pwc = frenetix.PoseWithCovariance(position, orientation, covariance_matrix)
                predicted_path[time_step] = pwc

            # Store the resulting predicted path
            self.predictionsForCpp[key] = frenetix.PredictedObject(int(key), predicted_path, pred['shape']['length'], pred['shape']['width'])

    def set_cost_function(self, cost_weights):
        self.config_plan.cost.cost_weights = cost_weights
        self.trajectory_handler_set_constant_cost_functions()
        self.trajectory_handler_set_constant_feasibility_functions()
        self.trajectory_handler_set_changing_functions()
        if self.logger:
            self.logger.set_logging_header(self.config_plan.cost.cost_weights)

    def trajectory_handler_set_constant_feasibility_functions(self):
        self.handler.add_feasability_function(ff.CheckYawRateConstraint(deltaMax=self.vehicle_params.delta_max,
                                                                        wheelbase=self.vehicle_params.wheelbase,
                                                                        wholeTrajectory=False
                                                                        ))
        self.handler.add_feasability_function(ff.CheckAccelerationConstraint(switchingVelocity=self.vehicle_params.v_switch,
                                                                             maxAcceleration=self.vehicle_params.a_max,
                                                                             wholeTrajectory=False)
                                                                             )
        self.handler.add_feasability_function(ff.CheckCurvatureConstraint(deltaMax=self.vehicle_params.delta_max,
                                                                          wheelbase=self.vehicle_params.wheelbase,
                                                                          wholeTrajectory=False
                                                                          ))
        self.handler.add_feasability_function(ff.CheckCurvatureRateConstraint(wheelbase=self.vehicle_params.wheelbase,
                                                                              velocityDeltaMax=self.vehicle_params.v_delta_max,
                                                                              wholeTrajectory=False
                                                                              ))

    def trajectory_handler_set_constant_cost_functions(self):
        name = "acceleration"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateAccelerationCost(name, self.cost_weights[name]))

        name = "jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateJerkCost(name, self.cost_weights[name]))

        name = "lateral_jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateLateralJerkCost(name, self.cost_weights[name]))

        name = "longitudinal_jerk"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateLongitudinalJerkCost(name, self.cost_weights[name]))

        name = "orientation_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateOrientationOffsetCost(name, self.cost_weights[name]))

        name = "lane_center_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateLaneCenterOffsetCost(name, self.cost_weights[name]))

        name = "distance_to_reference_path"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateDistanceToReferencePathCost(name, self.cost_weights[name]))

    def trajectory_handler_set_changing_functions(self):
        self.handler.add_function(frenetix.trajectory_functions.FillCoordinates(
            lowVelocityMode=self._LOW_VEL_MODE,
            initialOrientation=self.x_0.orientation,
            coordinateSystem=self.coordinate_system_cpp,
            horizon=int(self.config_plan.planning.planning_horizon)
        ))

        name = "prediction"
        if name in self.cost_weights.keys():
            self.handler.add_cost_function(
                cf.CalculateCollisionProbabilityFast(name, self.cost_weights[name], self.predictionsForCpp,
                                                     self.vehicle_params.length, self.vehicle_params.width, self.vehicle_params.wb_rear_axle))

        name = "distance_to_obstacles"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            obstacle_positions = np.zeros((len(self.scenario.obstacles), 2))
            for i, obstacle in enumerate(self.scenario.obstacles):
                state = obstacle.state_at_time(self.x_0.time_step)
                if state is not None:
                    obstacle_positions[i, 0] = state.position[0]
                    obstacle_positions[i, 1] = state.position[1]

            self.handler.add_cost_function(cf.CalculateDistanceToObstacleCost(name, self.cost_weights[name], obstacle_positions))

        name = "velocity_offset"
        if name in self.cost_weights.keys() and self.cost_weights[name] > 0:
            self.handler.add_cost_function(cf.CalculateVelocityOffsetCost(name, self.cost_weights[name], self.desired_velocity))

    def set_reference_and_coordinate_system(self, reference_path: np.ndarray):
        """
        Automatically creates a curvilinear coordinate system from a given reference path
        :param reference_path: reference_path as polyline
        """
        self.coordinate_system = CoordinateSystem(reference=reference_path, config_sim=self.config_sim)

        # For manual debugging reasons:
        if self.config_sim.visualization.ref_path_debug:
            visualize_scenario_and_pp(scenario=self.scenario, planning_problem=self.planning_problem,
                                      save_path=self.config_sim.simulation.log_path, cosy=self.coordinate_system)

        self.coordinate_system_cpp: frenetix.CoordinateSystemWrapper = frenetix.CoordinateSystemWrapper(reference_path)
        self.set_new_ref_path = True
        if self.logger:
            self.logger.sql_logger.write_reference_path(reference_path)

    def _get_cartesian_state(self) -> frenetix.CartesianPlannerState:
        return frenetix.CartesianPlannerState(self.x_0.position, self.x_0.orientation, self.x_0.velocity, self.x_0.acceleration, self.x_0.steering_angle)

    def _compute_standstill_trajectory(self) -> frenetix.TrajectorySample:
        """
        Computes a standstill trajectory if the vehicle is already at velocity 0
        :return: The TrajectorySample for a standstill trajectory
        """

        ps = frenetix.PlannerState(
            self._get_cartesian_state(),
            frenetix.CurvilinearPlannerState(self.x_cl[0], self.x_cl[1]),
            self.vehicle_params.wheelbase
        )

        return frenetix.TrajectorySample.compute_standstill_trajectory(self.coordinate_system_cpp, ps, self.dT, self.horizon)
    
    def compute_vehicle_localization(self) -> tuple:
        x_0_lon = None
        x_0_lat = None
        if self.x_cl is None:
            x_cl_new = frenetix.compute_initial_state( 
                coordinate_system=self.coordinate_system_cpp,
                x_0=self._get_cartesian_state(),
                wheelbase=self.vehicle_params.wheelbase,
                low_velocity_mode=self._LOW_VEL_MODE
            )

            x_0_lat = x_cl_new.x0_lat
            x_0_lon = x_cl_new.x0_lon
        else:
            x_0_lat = self.x_cl[1]
            x_0_lon = self.x_cl[0]
        return x_0_lat, x_0_lon
    
    def create_sampling_csv(self, dataframe: pd.DataFrame, filename: str, dense: bool):
        timestamp = time.strftime("%H")
        column_names = ['t0_range', 't1_range', 's0_range', 'ss0_range', 'sss0_range', 'ss1_range', 'sss1_range', 'd0_range', 'dd0_range', 'ddd0_range', 'd1_range', 'dd1_range', 'ddd1_range']
        if dense == True:
            dataframe.to_csv(f"frenetix_motion_planner/sampling_matrices/dense/{filename}_", mode='a', header=False, index=False)
        else:
            dataframe.to_csv(f"frenetix_motion_planner/sampling_matrices/sparse/{filename}_", mode='a', header=False, index=False)

    def find_factor_pairs(self, goal, tolerance):
        factor_pairs = []

        for i in range(1, goal + tolerance + 1):
            quotient = (goal + tolerance) // i
            product = i * quotient
            if goal - tolerance <= product <= goal + tolerance and quotient > 2 and i > 2:
                pair = [i, quotient]
                factor_pairs.append(pair)
                if pair[0] != pair[1]:
                    factor_pairs.append(pair[::-1])

        factor_pairs = [list(t) for t in set(tuple(p) for p in factor_pairs)]
        
        factor_pairs.sort(key=lambda x: abs(x[0] - x[1]))
        return factor_pairs
    
    def get_spacing(self, stage, tolerance):
        self.t1 = self.config_plan.planning.t1_len
        count = 0
        goal = int(self.num_trajectories[stage] / self.t1[stage])
        combinations = self.find_factor_pairs(goal, tolerance=tolerance)
        d1_values = []
        ss1_values = []
        for i in range(len(combinations)):
            d1_spacing = combinations[i][0]
            d1_values.append(d1_spacing)
            ss1_spacing = combinations[i][1]
            ss1_values.append(ss1_spacing)
            count += 1
        return d1_values, ss1_values, count
        
        
    def update_spacing(self, level, tolerance):
        d1_list = []
        ss1_list = []
        yaml_path = "configurations/frenetix_motion_planner/spacing.yaml"
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        for i in range(self.sampling_depth):
            d1, ss1, _ = self.get_spacing(i, tolerance)
            d1_list.append(d1[level])
            ss1_list.append(ss1[level])
        config['d1_values'] = d1_list
        config['ss1_values'] = ss1_list
        with open(yaml_path, 'w') as file:
            yaml.safe_dump(config, file)

    def get_optimal_sampling_window(self, optimal_parameters: list, width_factor: float) -> list:
        optimal_window = []
        optimal_ss1, optimal_d1 = optimal_parameters[5], optimal_parameters[-3]
        velocity_mult = 10
        optimal_window.append([optimal_ss1 - (width_factor * velocity_mult), optimal_ss1 + (width_factor * velocity_mult)])
        optimal_window.append([optimal_d1 - width_factor, optimal_d1 + width_factor])
        return optimal_window
    
    def initial_sampling_variables(self, level: int, longitude: tuple, latitude: tuple, optimal_window: list = None, mode : str = "normal") -> float:
        self.horizon = self.config_plan.planning.planning_horizon
        self.t_min = self.config_plan.planning.t_min
        self.spacing = self.config_plan.planning.spacing
        """
        Get the initial sampling variables for sparse and dense sampling windows
        :param dense_sampling boolean to determine if plan function is in the dense sampling phase
        :param optimal window: Values based around the optimal sampling parameters determined in the sparse stage 
        :return: New sampling parameters for sampling matrix generation
        """
        if level > 1:
            t1_range = np.array(list(self.sampling_handler.t_sampling.to_range(level, self.spacing, min_val=self.t_min, max_val= self.horizon, type='t1', mode=mode, user_input=self.user_input).union({self.N * self.dT})))
            ss1_range = np.array(list(self.sampling_handler.v_sampling.to_range(level, self.spacing, min_val=optimal_window[0][0], max_val=optimal_window[0][1], type='ss1', mode=mode, user_input=self.user_input).union({longitude[1]})))
            d1_range = np.array(list(self.sampling_handler.d_sampling.to_range(level, self.spacing, min_val=optimal_window[1][0], max_val=optimal_window[1][1], type='d1', mode=mode, user_input=self.user_input).union({latitude[0]})))
        else:
            t1_range = np.array(list(self.sampling_handler.t_sampling.to_range(level, self.spacing, type='t1', mode=mode, user_input=self.user_input).union({self.N*self.dT})))
            ss1_range = np.array(list(self.sampling_handler.v_sampling.to_range(level, self.spacing, type='ss1', mode=mode, user_input=self.user_input).union({longitude[1]})))
            d1_range = np.array(list(self.sampling_handler.d_sampling.to_range(level, self.spacing, type='d1', mode=mode, user_input=self.user_input).union({latitude[0]})))
        return t1_range, ss1_range, d1_range
    
    def get_feasibility(self, sampling_matrix) -> list:
        self.handler.generate_trajectories(sampling_matrix, self._LOW_VEL_MODE)

        if not self.config_plan.debug.multiproc or (self.config_sim.simulation.use_multiagent and
                                                    self.config_sim.simulation.multiprocessing):
            self.handler.evaluate_all_current_functions(True)
        else:
            self.handler.evaluate_all_current_functions_concurrent(True)

        feasible_trajectories = []
        infeasible_trajectories = []
        for trajectory in self.handler.get_sorted_trajectories():
            if trajectory.feasible:
                feasible_trajectories.append(trajectory)
            elif trajectory.valid:
                infeasible_trajectories.append(trajectory)

        if len(feasible_trajectories) + len(infeasible_trajectories) < 1:
            self.msg_logger.critical("No Valid Trajectories!")
        else:
            self.infeasible_kinematics_percentage = float(len(feasible_trajectories)
                                                    / (len(feasible_trajectories) + len(infeasible_trajectories))) * 100

        self.msg_logger.debug('Found {} feasible trajectories and {} infeasible trajectories'.format(feasible_trajectories.__len__(), infeasible_trajectories.__len__()))
        self.msg_logger.debug(
            'Percentage of valid & feasible trajectories: %s %%' % str(self.infeasible_kinematics_percentage))
        return feasible_trajectories, infeasible_trajectories
    
    def get_sampling_matrix(self, sampling_variables, x_0_lat, x_0_lon):
        t1_range = sampling_variables[0]
        ss1_range = sampling_variables[1]
        d1_range = sampling_variables[2]
        sampling_matrix = generate_sampling_matrix(t0_range = 0,
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
        return sampling_matrix
    
    def get_alternate_traj(self, optimal_trajectory, feasible_trajectories, sampling_matrix, lat):
        if optimal_trajectory is None and feasible_trajectories:
            if self.config_plan.planning.emergency_mode == "stopping":
                alt_optimal_trajectory = self._select_stopping_trajectory(feasible_trajectories, sampling_matrix, lat[0])
                self.msg_logger.warning("No optimal trajectory available. Select stopping trajectory!")
            else:
                for traje in feasible_trajectories:
                    self.set_risk_costs(traje)
                sort_risk = sorted(feasible_trajectories, key=lambda traj: traj._ego_risk + traj._obst_risk, reverse=False)
                self.msg_logger.warning("No optimal trajectory available. Select lowest risk trajectory!")
                alt_optimal_trajectory = sort_risk[0]
            return alt_optimal_trajectory
        return None



    def plan(self) -> tuple:
        """
        Plans an optimal trajectory
        :return: Optimal trajectory as tuple
        """
        self._infeasible_count_kinematics = np.zeros(11)
        self._collision_counter = 0
        self.infeasible_kinematics_percentage = 0
        self.samp_level = 0
        self.tolerance = 0
        self.mode = ''
        self.optimal_trajectories = []
        self.sampling_depth = self.config_plan.planning.sampling_depth
        # **************************************
        # Initialization of Cpp Frenet Functions
        # **************************************
        self.trajectory_handler_set_changing_functions()

        # **************************************
        # Vehicle Localization
        # **************************************
        x_0_lat, x_0_lon = self.compute_vehicle_localization()

        self.msg_logger.debug('Initial state is: lon = {} / lat = {}'.format(x_0_lon, x_0_lat))
        self.msg_logger.debug('Desired velocity is {} m/s'.format(self.desired_velocity))


        optimal_trajectory = None
        sampling_matrix = None
        t0 = time.time()

        optimal_parameters = None
        window_size = 0.5

        while len(self.optimal_trajectories) != self.sampling_depth:
            # *************************************
            # Create a sampling window around sampling optimal lateral displacement
            # *************************************
            i = 0
            while i < self.sampling_depth:
                level = (i + 1)
                if level > 1 and len(self.optimal_trajectories) == i and self.optimal_trajectories[i - 1] != None:
                    self.msg_logger.info(f"Sampling Stage: {level}")
                    optimal_parameters = getattr(self.optimal_trajectories[i - 1], 'sampling_parameters')
                    window_size = window_size * self.width_factor
                    sampling_window = self.get_optimal_sampling_window(optimal_parameters, window_size)
                    self.msg_logger.info(f"Sampling being done around {window_size} of the previous Optimal Sampling Parameters")

                    if self.mode == 'emergency':
                        test_sampling_variables = self.initial_sampling_variables(level, x_0_lon, x_0_lat, optimal_window=sampling_window, mode='emergency')
                    else:
                        test_sampling_variables = self.initial_sampling_variables(level, x_0_lon, x_0_lat, optimal_window=sampling_window)

                    sampling_matrix = self.get_sampling_matrix(test_sampling_variables, x_0_lat, x_0_lon)
                        
                    self.handler.reset_Trajectories()
                    feasible_trajectories, infeasible_trajectories = self.get_feasibility(sampling_matrix)
                    optimal_trajectory = self.trajectory_collision_check(feasible_trajectories)
                    
                    if self.user_input == [False, True]:
                        _, _, combos = self.get_spacing(i, self.tolerance)
                    if optimal_trajectory is None and self.samp_level < (combos - 1) and self.user_input == [False, True]:
                        self.mode = 'normal'
                        self.samp_level += 1
                        self.update_spacing(self.samp_level, self.tolerance)
                        self.msg_logger.info(f"Selecting from {sampling_matrix.shape[0]} trajectories")
                        continue

                    elif optimal_trajectory is None and feasible_trajectories == []:
                        self.tolerance *= 2
                        self.samp_level = 0
                        self.mode = 'emergency'
                        continue

                    if optimal_trajectory is not None:
                        self.msg_logger.info(f"Selecting from {sampling_matrix.shape[0]} trajectories")
                        self.optimal_trajectories.append(optimal_trajectory)
                        i += 1

                    elif optimal_trajectory is None and self.x_0.velocity <= 0.1:
                        optimal_trajectory = self._compute_standstill_trajectory()
                        self.optimal_trajectories.append(optimal_trajectory)
                        i += 1
                    else:
                        optimal_trajectory = self.get_alternate_traj(optimal_trajectory, feasible_trajectories, sampling_matrix, x_0_lat)
                        self.optimal_trajectories.append(optimal_trajectory)
                        i += 1
                                
    
                else:
                    self.msg_logger.info(f"Sampling Stage: 1")
                    
                    if self.mode == 'emergency':
                        test_sampling_variables = self.initial_sampling_variables(level, x_0_lon, x_0_lat, mode='emergency')
                    else:
                        test_sampling_variables = self.initial_sampling_variables(level, x_0_lon, x_0_lat)
                    sampling_matrix = self.get_sampling_matrix(test_sampling_variables, x_0_lat, x_0_lon)
                    self.handler.reset_Trajectories()
                    feasible_trajectories, infeasible_trajectories = self.get_feasibility(sampling_matrix)

                    optimal_trajectory = self.trajectory_collision_check(feasible_trajectories)

                    _, _, combos = self.get_spacing(i, self.tolerance)
                    if optimal_trajectory is None and self.samp_level < (combos - 1):
                        self.mode = 'normal'
                        self.samp_level += 1
                        self.update_spacing(self.samp_level, self.tolerance)
                        self.msg_logger.info(f"Selecting from {sampling_matrix.shape[0]} trajectories")
                        continue

                    elif optimal_trajectory is None and feasible_trajectories == []:
                        self.tolerance *= 2
                        self.samp_level = 0
                        self.mode = 'emergency'
                        continue
                    
                    elif optimal_trajectory is not None:
                        self.msg_logger.info(f"Selecting from {sampling_matrix.shape[0]} trajectories")
                        self.optimal_trajectories.append(optimal_trajectory)
                        i += 1
                    
                    elif optimal_trajectory is None and self.x_0.velocity <= 0.1:
                        optimal_trajectory = self._compute_standstill_trajectory()
                        self.optimal_trajectories.append(optimal_trajectory)
                        i += 1
                    else:
                        optimal_trajectory = self.get_alternate_traj(optimal_trajectory, feasible_trajectories, sampling_matrix, x_0_lat)
                        self.optimal_trajectories.append(optimal_trajectory)
                        i += 1



        planning_time = time.time() - t0
        self.total_time += planning_time
        self.count += 1
        print(self.count)
        print(f" AVERAGE TIME: {self.total_time / self.count}")

        self.transfer_infeasible_logging_information(infeasible_trajectories)

        self.msg_logger.debug('Rejected {} infeasible trajectories due to kinematics'.format(
            self._infeasible_count_kinematics))
        self.msg_logger.debug('Rejected {} infeasible trajectories due to collisions'.format(
            self.infeasible_count_collision))

        # ******************************************
        # Update Trajectory Pair & Commonroad Object
        # ******************************************
        self.trajectory_pair = self._compute_trajectory_pair(self.optimal_trajectories[-1]) if self.optimal_trajectories[-1] is not None else None
        if self.trajectory_pair is not None:
            current_ego_vehicle = self.convert_state_list_to_commonroad_object(self.trajectory_pair[0].state_list,
                                                                               self.config_sim.simulation.ego_agent_id)
            self.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

        # ************************************
        # Set Risk Costs to Optimal Trajectory
        # ************************************
        if self.optimal_trajectories[-1] is not None and self.log_risk:
            optimal_trajectory = self.set_risk_costs(self.optimal_trajectories[-1])

        self.optimal_trajectory = optimal_trajectory

        # **************************
        # Logging
        # **************************
        # for visualization store all trajectories with validity level based on kinematic validity
        if self._draw_traj_set or self.save_all_traj:
            self.all_traj = feasible_trajectories + infeasible_trajectories

        self.plan_postprocessing(optimal_trajectory=optimal_trajectory, planning_time=planning_time)
        return self.trajectory_pair

    @staticmethod
    def _select_stopping_trajectory(trajectories, sampling_matrix, d_pos):

        min_v_list = np.unique(sampling_matrix[:, 5])
        min_t_list = np.unique(sampling_matrix[:, 1])

        min_d_list = np.unique(sampling_matrix[:, 10])
        sorted_d_indices = np.argsort(np.abs(min_d_list - d_pos))
        min_d_list = min_d_list[sorted_d_indices]

        # Create a dictionary for quick lookups
        trajectory_dict = {}
        for traj in trajectories:
            v, t, d = traj.sampling_parameters[5], traj.sampling_parameters[1], traj.sampling_parameters[10]
            if v not in trajectory_dict:
                trajectory_dict[v] = {}
            if t not in trajectory_dict[v]:
                trajectory_dict[v][t] = {}
            trajectory_dict[v][t][d] = traj

        # Check combinations of v, t, d values for valid trajectories
        for v, t, d in product(min_v_list, min_t_list, min_d_list):
            if v in trajectory_dict and t in trajectory_dict[v] and d in trajectory_dict[v][t]:
                return trajectory_dict[v][t][d]

    def transfer_infeasible_logging_information(self, infeasible_trajectories):

        feas_list = [i.feasabilityMap['Curvature Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self._infeasible_count_kinematics[5] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Yaw rate Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self._infeasible_count_kinematics[6] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Curvature Rate Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self._infeasible_count_kinematics[7] = int(sum(acc_feas))

        feas_list = [i.feasabilityMap['Acceleration Constraint'] for i in infeasible_trajectories]
        acc_feas = [int(1) if num > 0 else int(0) for num in feas_list]
        self._infeasible_count_kinematics[8] = int(sum(acc_feas))

        self._infeasible_count_kinematics[0] = int(sum(self._infeasible_count_kinematics))