import os
from pathlib import Path
from typing import Union
from omegaconf import OmegaConf, ListConfig, DictConfig

# commonroad-io
from commonroad.common.solution import VehicleType

# commonroad-dc
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping


class Configuration:
    """
    Main Configuration class holding all planner-relevant configurations
    """
    def __init__(self, config: Union[ListConfig, DictConfig]):
        # initialize subclasses
        self.multiagent: MultiagentConfiguration = MultiagentConfiguration(config.multiagent)
        self.planning: PlanningConfiguration = PlanningConfiguration(config.planning)
        self.prediction: PredictionConfiguration = PredictionConfiguration(config.prediction)
        self.vehicle: VehicleConfiguration = VehicleConfiguration(config.vehicle)
        self.sampling: SamplingConfiguration = SamplingConfiguration(config.sampling)
        self.debug: DebugConfiguration = DebugConfiguration(config.debug)
        self.general: GeneralConfiguration = GeneralConfiguration(config.general)
        self.cost: CostConfiguration = CostConfiguration(config.cost)
        self.occlusion: OcclusionModuleConfiguration = OcclusionModuleConfiguration(config.occlusion)
        self.behavior: BehaviorPlannerConfiguration = BehaviorPlannerConfiguration(config.behaviorplanner)


class BehaviorPlannerConfiguration:
    """Class to store additional configurations for multiagent simulations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.use_behavior_planner = config.use_behavior_planner


class MultiagentConfiguration:
    """Class to store additional configurations for multiagent simulations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.agent_ids = config.agent_ids
        self.show_individual_plots = config.show_individual_plots
        self.save_individual_plots = config.save_individual_plots
        self.save_individual_gifs = config.save_individual_gifs
        self.multiprocessing = config.multiprocessing
        self.num_procs = config.num_procs


class PlanningConfiguration:
    """Class to store all planning configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.dt = config.dt
        self.time_steps_computation = config.time_steps_computation
        self.planning_horizon = config.dt * config.time_steps_computation
        self.replanning_frequency = config.replanning_frequency
        self.mode = config.mode
        self.continuous_cc = config.continuous_cc
        self.collision_check_in_cl = config.collision_check_in_cl
        self.factor = config.factor
        self.low_vel_mode_threshold = config.low_vel_mode_threshold
        self.use_clusters = config.use_clusters
        self.cluster_means = config.cluster_means
        self.cluster_stds = config.cluster_stds


class PredictionConfiguration:
    """Class to store all prediction configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.mode = config.mode
        self.calc_visible_area = config.calc_visible_area
        self.sensor_radius = config.sensor_radius
        self.cone_angle = config.cone_angle
        self.cone_safety_dist = config.cone_safety_dist
        self.pred_horizon_in_s = config.pred_horizon_in_s


class VehicleConfiguration:
    """Class to store vehicle configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.cr_vehicle_id = config.cr_vehicle_id

        # get vehicle parameters from CommonRoad vehicle models given cr_vehicle_id
        vehicle_parameters = VehicleParameterMapping.from_vehicle_type(VehicleType(config.cr_vehicle_id))

        # get dimensions from given vehicle ID
        self.length = vehicle_parameters.l
        self.width = vehicle_parameters.w
        self.front_ax_distance = vehicle_parameters.a
        self.rear_ax_distance = vehicle_parameters.b
        self.wheelbase = vehicle_parameters.a + vehicle_parameters.b
        self.mass = vehicle_parameters.m

        # get constraints from given vehicle ID
        self.a_max = vehicle_parameters.longitudinal.a_max
        self.v_max = vehicle_parameters.longitudinal.v_max
        self.v_switch = vehicle_parameters.longitudinal.v_switch
        self.delta_min = vehicle_parameters.steering.min
        self.delta_max = vehicle_parameters.steering.max
        self.v_delta_min = vehicle_parameters.steering.v_min
        self.v_delta_max = vehicle_parameters.steering.v_max

        # overwrite parameters given by vehicle ID if they are explicitly provided in the *.yaml file
        for key, value in config.items():
            if value is not None:
                setattr(self, key, value)


class SamplingConfiguration:
    """Class to store sampling configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.t_min = config.t_min
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.d_min = config.d_min
        self.d_max = config.d_max
        self.sampling_min = config.sampling_min
        self.sampling_max = config.sampling_max


class DebugConfiguration:
    """Class to store debug configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.save_all_traj = config.save_all_traj
        self.save_unweighted_costs = config.save_unweighted_costs
        self.log_risk = config.log_risk
        self.show_plots = config.show_plots
        self.save_plots = config.save_plots
        self.evaluation = config.evaluation
        self.collision_report = config.collision_report
        self.gif = config.gif
        self.plot_window_dyn = config.plot_window_dyn
        self.draw_icons = config.draw_icons
        self.draw_traj_set = config.draw_traj_set
        self.debug_mode = config.debug_mode
        self.multiproc = config.multiproc
        self.num_workers = config.num_workers
        self.kinematic_debug = config.kinematic_debug


class GeneralConfiguration:
    def __init__(self, config: Union[ListConfig, DictConfig]):
        name_scenario = config.name_scenario

        self.path_scenarios = config.path_scenarios
        self.path_scenario = name_scenario  # config.path_scenarios + name_scenario + ".xml"
        self.path_output = config.path_output
        self.max_steps = config.max_steps


class CostConfiguration:
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.params = config.params


class OcclusionModuleConfiguration:
    """Class to store all occlusion configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.use_occlusion_module = config.use_occlusion_module
        self.scope = config.scope
        self.visibility_estimation = config.visibility_estimation
        self.trajectory_eval = config.trajectory_eval
        self.use_phantom_ped = config.use_phantom_ped
        self.create_commonroad_obstacle = config.create_commonroad_obstacle
        self.collision_check_mode = config.collision_check_mode
        self.visualize_collision = config.visualize_collision
        self.show_occlusion_plot = config.show_occlusion_plot
        self.save_plot = config.save_plot
        self.interactive_plot = config.interactive_plot
        self.plot_backend = config.plot_backend
        self.use_fast_plot = config.use_fast_plot
        self.plot_window_dyn = config.plot_window_dyn
