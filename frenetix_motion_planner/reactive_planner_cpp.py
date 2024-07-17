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

# frenetix_motion_planner imports
from frenetix_motion_planner.sampling_matrix import generate_sampling_matrix
from frenetix_motion_planner.sampling_matrix import SamplingHandler

from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem
from cr_scenario_handler.utils.visualization import visualize_scenario_and_pp

import frenetix
import frenetix.trajectory_functions
import frenetix.trajectory_functions.feasability_functions as ff
import frenetix.trajectory_functions.cost_functions as cf

from frenetix_motion_planner.planner import Planner
from config import SCENARIO_NAME


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
        self.trajectory_handler_set_constant_cost_functions()
        self.trajectory_handler_set_constant_feasibility_functions()
        self.window_size = 1
        self.num_points = 10

    def set_predictions(self, predictions: dict):
        self.use_prediction = True
        self.predictions = predictions
        for key, pred in self.predictions.items():
            num_steps = pred['pos_list'].shape[0] #taking the amount of available predictions as num_of steps
            predicted_path: List[frenetix.PoseWithCovariance] = [None] * num_steps #makes a list filled with none num_steps times.

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
        samp_level = self._sampling_min
        window_size = self.window_size
        points = self.num_points
        timestamp = time.strftime("%H")
        column_names = ['t0_range', 't1_range', 's0_range', 'ss0_range', 'sss0_range', 'ss1_range', 'sss1_range', 'd0_range', 'dd0_range', 'ddd0_range', 'd1_range', 'dd1_range', 'ddd1_range']
        if dense == True:
            dataframe.to_csv(f"frenetix_motion_planner/sampling_matrices/dense/{filename}_{samp_level}", mode='a', header=False, index=False)
        else:
            dataframe.to_csv(f"frenetix_motion_planner/sampling_matrices/sparse/{filename}_{samp_level}", mode='a', header=False, index=False)
    
    def initial_sampling_variables(self, samp_level: int, longitude: tuple, latitude: tuple, dense_sampling: bool = False) -> float:
        t1_range = np.array(list(self.sampling_handler.t_sampling.to_range(samp_level, dense_sampling).union({self.N*self.dT})))
        ss1_range = np.array(list(self.sampling_handler.v_sampling.to_range(samp_level, dense_sampling).union({longitude[1]}))) # where min / max is set for v sampling window
        d1_range = np.array(list(self.sampling_handler.d_sampling.to_range(samp_level, dense_sampling).union({latitude[0]})))
        return t1_range, ss1_range, d1_range
    
    def get_sampling_window(self, optimal_parameter: list, window_size: int, num_points: int) -> list:
        """
        Adjust the sampling window around the optimal lateral displacement.
        :param optimal_lat_displacement: Optimal lateral displacement found in sparse sampling.
        :param window_size: The range around the optimal displacement to sample within.
        :param num_points: Number of points to sample within the window.
        :return: New d1_range for dense sampling.
        """
        optimal_value = float(optimal_parameter)
        sampling_window = np.linspace(optimal_value - window_size, optimal_value + window_size, num=num_points)
        return sampling_window
    
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


    def plan(self) -> tuple:
        """
        Plans an optimal trajectory
        :return: Optimal trajectory as tuple
        """
        self._infeasible_count_kinematics = np.zeros(11)
        self._collision_counter = 0
        self.infeasible_kinematics_percentage = 0
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

        samp_level = self._sampling_min
        dense_sampling = False

        while True:
            t1_range, ss1_range, d1_range = self.initial_sampling_variables(samp_level, x_0_lon, x_0_lat, dense_sampling=False)
            print(t1_range)
            print(ss1_range)
            print(d1_range)
            # *************************************
            # Create a sampling window around sparse sampling optimal lateral displacement
            # *************************************
            if dense_sampling:
                t1_range, ss1_range, d1_range = self.initial_sampling_variables(samp_level, x_0_lon, x_0_lat, dense_sampling=True)
                # all you have to do here is increase samp level while decreasing the "sampling window"
            
            sampling_matrix = generate_sampling_matrix(t0_range=0.0, #initial time
                                                       t1_range=t1_range, #final time
                                                       s0_range=x_0_lon[0], #specific longitudinal position
                                                       ss0_range=x_0_lon[1], #initial longitudinal velocity
                                                       sss0_range=x_0_lon[2], #initial longitudinal acceleration
                                                       ss1_range=ss1_range, #possible velocity changes
                                                       sss1_range=0, #possible acceleration changes
                                                       d0_range=x_0_lat[0], #inital lateral displacement
                                                       dd0_range=x_0_lat[1], #initial lateral velocity
                                                       ddd0_range=x_0_lat[2], #initial lateral acceleration
                                                       d1_range=d1_range, #lateral state displacement
                                                       dd1_range=0.0, #lateral state velocity (derivative) -- 0 because we want to be parallel with the reference path
                                                       ddd1_range=0.0) #lateral state acceleration (derivative)
            
            print(f"Matrix is {sampling_matrix.shape} AND Samp Min is: {samp_level}")
            
            if dense_sampling:
                df = pd.DataFrame(sampling_matrix)
                self.create_sampling_csv(df, f"{SCENARIO_NAME}", dense_sampling)
            else:
                df = pd.DataFrame(sampling_matrix)
                self.create_sampling_csv(df, f"{SCENARIO_NAME}", dense_sampling)

            self.handler.reset_Trajectories()
            feasible_trajectories, infeasible_trajectories = self.get_feasibility(sampling_matrix)

            # ******************************************
            # Check Feasible Trajectories for Collisions
            # ******************************************

            optimal_trajectory = self.trajectory_collision_check(feasible_trajectories)
            #needs to be outside of the loop in case an optimal trajectory isn't found first time through it is None.
            if optimal_trajectory != None:
                optimal_parameters = getattr(optimal_trajectory, 'sampling_parameters')

            if dense_sampling == True:
                reshaped_optimal_parameters = np.array(optimal_parameters).reshape(1, 13)
                optimal_dense_df = pd.DataFrame(reshaped_optimal_parameters)
                self.create_sampling_csv(optimal_dense_df, f"optimal_{SCENARIO_NAME}", dense_sampling)
            reshaped_optimal_parameters = np.array(optimal_parameters).reshape(1, 13)
            optimal_sparse_df = pd.DataFrame(reshaped_optimal_parameters)
            self.create_sampling_csv(optimal_sparse_df, f"optimal_{SCENARIO_NAME}", dense_sampling)

            # ******************************************
            # Check if Dense Layer has been Computed
            # ******************************************

            if optimal_trajectory is not None and not dense_sampling:
                dense_sampling = True
                samp_level += 1
                continue
            elif dense_sampling:
                break
            else:
                samp_level += 1


        planning_time = time.time() - t0

        self.transfer_infeasible_logging_information(infeasible_trajectories)

        self.msg_logger.debug('Rejected {} infeasible trajectories due to kinematics'.format(
            self._infeasible_count_kinematics))
        self.msg_logger.debug('Rejected {} infeasible trajectories due to collisions'.format(
            self.infeasible_count_collision))

        # ************************************************
        # Fall back to standstill trajectory if applicable
        # ************************************************
        if optimal_trajectory is None and self.x_0.velocity <= 0.1:
            self.msg_logger.warning('Planning standstill for the current scenario')
            if self.logger:
                self.logger.trajectory_number = self.x_0.time_step
            optimal_trajectory = self._compute_standstill_trajectory()

        # *******************************************
        # Find alternative Optimal Trajectory if None
        # *******************************************
        if optimal_trajectory is None and feasible_trajectories:
            if self.config_plan.planning.emergency_mode == "stopping":
                optimal_trajectory = self._select_stopping_trajectory(feasible_trajectories, sampling_matrix, x_0_lat[0])
                self.msg_logger.warning("No optimal trajectory available. Select stopping trajectory!")
            else:
                for traje in feasible_trajectories:
                    self.set_risk_costs(traje)
                sort_risk = sorted(feasible_trajectories, key=lambda traj: traj._ego_risk + traj._obst_risk, reverse=False)
                self.msg_logger.warning("No optimal trajectory available. Select lowest risk trajectory!")
                optimal_trajectory = sort_risk[0]

        # ******************************************
        # Update Trajectory Pair & Commonroad Object
        # ******************************************
        self.trajectory_pair = self._compute_trajectory_pair(optimal_trajectory) if optimal_trajectory is not None else None
        if self.trajectory_pair is not None:
            current_ego_vehicle = self.convert_state_list_to_commonroad_object(self.trajectory_pair[0].state_list,
                                                                               self.config_sim.simulation.ego_agent_id)
            self.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

        # ************************************
        # Set Risk Costs to Optimal Trajectory
        # ************************************
        if optimal_trajectory is not None and self.log_risk:
            optimal_trajectory = self.set_risk_costs(optimal_trajectory)

        self.optimal_trajectory = optimal_trajectory

        # **************************
        # Logging
        # **************************
        # for visualization store all trajectories with validity level based on kinematic validity
        if self._draw_traj_set or self.save_all_traj:
            self.all_traj = feasible_trajectories + infeasible_trajectories

        self.plan_postprocessing(optimal_trajectory=optimal_trajectory, planning_time=planning_time)

        return self.trajectory_pair
        #frenet interface is where is decisions are made on which trajectory to use

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
