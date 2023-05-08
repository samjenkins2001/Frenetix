"""
This module is the main module of the Occlusion calculation of the reactive planner.
All submodules are started and managed from here

Author: Korbinian Moller, TUM
Date: 27.04.2023
"""

# imports
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
from functools import reduce
from omegaconf import OmegaConf

# commonroad inputs
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as ohf
from commonroad_rp.occlusion_planning.utils.visualization import OccPlot
from commonroad_rp.occlusion_planning.basic_modules.visibility_module import VisibilityModule
from commonroad_rp.occlusion_planning.basic_modules.uncertainty_map import OccUncertaintyMap
from commonroad_rp.occlusion_planning.evaluation_modules.visibility_estimator import OccVisibilityEstimator
from commonroad_rp.occlusion_planning.evaluation_modules.uncertainty_map_evaluator import OccUncertaintyMapEvaluator
from commonroad_rp.occlusion_planning.evaluation_modules.phantom_module import OccPhantomModule


class OcclusionModule:
    def __init__(self, scenario, config, ref_path, log_path, planner):
        self.scenario_id = scenario.scenario_id
        self.occ_config = config.occlusion
        self.cost_config = OmegaConf.to_object(config.cost.params)
        self.log_path = log_path
        self.predictions = None
        self.debug_mode = config.debug.debug_mode
        self.plot = config.occlusion.show_occlusion_plot
        if config.occlusion.scope == "sensor_radius":
            self.scope = config.prediction.sensor_radius
        else:
            self.scope = config.occlusion.scope

        # initialize occ scenario which stores the reference path and shapely polygons of the lanelet network
        self.occ_scenario = OccScenario(ego_width=config.vehicle.width,
                                        ego_length=config.vehicle.length,
                                        ref_path=ref_path,
                                        scenario_lanelet_network=scenario.lanelet_network,
                                        dt=config.planning.dt,
                                        sidewalk_buffer=2)  # if sidewalk buffer > 0 sidewalks will be considered

        # initialize the visibility module which calculates the visible area (step is in prediction_preprocessing)
        # stores information about the visible area, occlusion obstacles, the ego position and the current timestep
        self.vis_module = VisibilityModule(scenario=scenario,
                                           lanelets=self.occ_scenario.lanelet_polygon,
                                           sensor_radius=config.prediction.sensor_radius)

        # initialize objects of class OccArea for visible and occluded areas storing shapely polygons and a point cloud
        self.occ_visible_area = OccArea(area_type="visible")
        self.occ_occluded_area = OccArea(area_type="occluded")

        # initialize the uncertainty map which contains information about occluded areas and their history
        self.occ_uncertainty_map = OccUncertaintyMap(debug_mode=config.debug.debug_mode,
                                                     occ_visible_area=self.occ_visible_area,
                                                     occ_occluded_area=self.occ_occluded_area)

        # if results of the occlusion module shall be plotted, an occlusion plot is initialized
        # (needed in further modules)
        if self.plot:
            self.occ_plot = OccPlot(config=config, log_path=self.log_path,
                                    scenario_id=self.scenario_id, occ_scenario=self.occ_scenario)
        else:
            self.occ_plot = None

        # initialize the phantom module which is used to predict phantom obstacles (pedestrians) and to evaluate
        # trajectories by their collision risk (harm_module)
        self.occ_phantom_module = OccPhantomModule(config=config,
                                                   occ_scenario=self.occ_scenario,
                                                   vis_module=self.vis_module,
                                                   occ_visible_area=self.occ_visible_area,
                                                   occ_plot=self.occ_plot,
                                                   params_risk=planner.params_risk,
                                                   params_harm=planner.params_harm,
                                                   debug_mode=self.debug_mode)

        # initialize the uncertainty map evaluator which evaluates trajectories in the uncertainty map
        self.occ_uncertainty_map_evaluator = OccUncertaintyMapEvaluator(self.vis_module, self.occ_uncertainty_map,
                                                                        self.occ_plot)

        # initialize the visibility estimator which evaluates trajectories according to their lateral position and the
        # corresponding visible area (calculates v_ratio and evaluates it)
        self.occ_visibility_estimator = OccVisibilityEstimator(self.occ_scenario, self.vis_module, self.occ_plot)

    def step(self, predictions=None):

        # calc visible and occluded area for further processing
        if self.vis_module.visible_area_timestep is not None:
            visible_area, occluded_area, add_occ_plot = ohf.calc_occluded_areas(
                    ego_pos=self.vis_module.ego_pos,
                    visible_area=self.vis_module.visible_area_timestep,
                    ref_path=self.occ_scenario.ref_path,
                    lanelets=self.occ_scenario.lanelet_polygon,
                    scope=self.scope,
                    ref_path_buffer=10,
                    cut_backwards=True)
        else:
            visible_area = None
            occluded_area = None
            add_occ_plot = None

        # store predictions
        self.predictions = predictions

        # set visible and occluded area
        self.occ_visible_area.set_area(visible_area)
        self.occ_occluded_area.set_area(occluded_area)

        # if plot is activated plot the oclusion scenario
        if self.plot:
            self.occ_plot.step_plot(time_step=self.vis_module.time_step,
                                    ego_state=self.vis_module.ego_state,
                                    ref_path=self.occ_scenario.ref_path,
                                    lanelet_polygon=self.occ_scenario.lanelets_single,
                                    sidewalk_polygon=self.occ_scenario.sidewalk_combined,
                                    lanelet_polygon_along_path=self.occ_scenario.lanelets_along_path_combined,
                                    # visible_area_vm=self.vis_module.visible_area_timestep,
                                    obstacles=self.vis_module.obstacles,
                                    obstacle_id=True,
                                    visible_area=self.occ_visible_area.poly,
                                    occluded_area=self.occ_occluded_area.poly,
                                    # additional_plot=add_occ_plot
                                    )

    def calc_costs(self, trajectories=None, scenario="cluster0", plot=True):
        # quit if error
        if trajectories is None or len(trajectories) == 0:
            raise ValueError('Trajectory list is empty')

        # evaluate phantom module if activated
        if self.occ_config.use_phantom_module:
            phantom_costs = self.occ_phantom_module.evaluate_trajectories(trajectories, max_harm=0.5, plot=False)
        else:
            phantom_costs = np.zeros(len(trajectories))

        # evaluate uncertainty map if activated
        if self.occ_config.use_uncertainty_map_evaluator:
            uncertainty_costs = self.occ_uncertainty_map_evaluator.evaluate_trajectories(trajectories,
                                                                                         plot_uncertainty_map=False,
                                                                                         plot=False)
        else:
            uncertainty_costs = np.zeros(len(trajectories))

        # evaluate visibility estimator if activated
        if self.occ_config.use_visibility_estimator and self.predictions is not None:
            vis_est_costs = self.occ_visibility_estimator.evaluate_trajectories(trajectories, self.predictions,
                                                                                plot_traj=True, plot_map=False)
        else:
            vis_est_costs = np.zeros(len(trajectories))

        # store costs in numpy array for multiplication with weights
        cost_list = np.array([phantom_costs, uncertainty_costs, vis_est_costs]).T

        # load weights from config and store in numpy array for multiplication
        weights = np.array([self.cost_config[scenario]["Occ_PM"],
                            self.cost_config[scenario]["Occ_UM"],
                            self.cost_config[scenario]["Occ_VE"]])

        # calc weighted cost list
        cost_list_weighted = cost_list * weights.reshape(1, -1)

        # calc sum
        total_cost = np.sum(cost_list_weighted, axis=1)

        # plot costs if plot is activated
        if self.occ_plot is not None and plot:
            self.occ_plot.plot_trajectories_cost_color(trajectories, total_cost)

        # calc new costs (traj cost + occlusion cost)
        for i, traj in enumerate(trajectories):
            traj_cost = traj.cost + total_cost[i]
            traj_cost_list = traj.cost_list

            # update trajectory costs
            traj.set_costs(traj_cost, np.append(traj_cost_list, cost_list_weighted[i]))


class OccScenario:
    def __init__(self, ego_width=1.8, ego_length=5, ref_path=None,
                 scenario_lanelet_network=None, dt=0.1, sidewalk_buffer=0):
        # private attributes
        self._scenario_lanelet_network = scenario_lanelet_network
        self._lanelets_polygon_list = None
        self._sidewalk_buffer = sidewalk_buffer

        # public attributes
        self.dt = dt
        self.ego_width = ego_width
        self.ego_length = ego_length
        self.ref_path = ref_path
        self.ref_path_ls = LineString(self.ref_path)
        # one polygon of the entire lanelet network
        self.lanelets_combined = None
        # one polygon of the entire lanelet network with sidewalk
        self.lanelets_combined_with_sidewalk = None
        # one polygon of the entire sidewalk (buffer around lanelets_combined)
        self.sidewalk_combined = None
        # multipolygon of the lanelets (each lanelet is one polygon)
        self.lanelets_single = None
        # one polygon of the lanelets along the reference path
        self.lanelets_along_path_combined = None
        # multipolygon of the lanelets along the reference path (each lanelet is one polygon)
        self.lanelets_along_path_single = None

        # function calls
        self._convert_lanelet_network()
        self._lanelets_along_path()

        # defines the lanelet polygon that is used for further calculation (in visibility and occlusion module)
        if sidewalk_buffer > 0:
            self.lanelet_polygon = self.lanelets_combined_with_sidewalk
        else:
            self.lanelet_polygon = self.lanelets_combined

    def _convert_lanelet_network(self):

        self._lanelets_polygon_list = [poly.shapely_object for poly in self._scenario_lanelet_network.lanelet_polygons]

        self.lanelets_combined = reduce(lambda x, y: x.union(y), self._lanelets_polygon_list)

        self.lanelets_combined_with_sidewalk = self.lanelets_combined.buffer(self._sidewalk_buffer)

        self.sidewalk_combined = self.lanelets_combined_with_sidewalk.difference(self.lanelets_combined)

        self.lanelets_single = ohf.convert_list_to_multipolygon(self._lanelets_polygon_list)

    def _lanelets_along_path(self):

        lanelets_along_path = []
        for poly in self._lanelets_polygon_list:
            intersection = poly.intersects(self.ref_path_ls)
            if intersection:
                lanelets_along_path.append(poly)

        self.lanelets_along_path_combined = unary_union(lanelets_along_path)
        self.lanelets_along_path_single = ohf.convert_list_to_multipolygon(lanelets_along_path)

    def set_ref_path(self, ref_path):
        self.ref_path = ref_path


class OccArea:
    def __init__(self, area_type=None):
        self.type = area_type
        self.poly = None
        self.area = None
        self.centroids = None
        self.points = None

    def set_area(self, polygon):
        if polygon is not None:
            self.poly = polygon
            self.area = polygon.area
            self.centroids = polygon.centroid
            self.points = ohf.point_cloud(polygon)
        else:
            self.poly = None
            self.area = None
            self.centroids = None
            self.points = None
# eof
