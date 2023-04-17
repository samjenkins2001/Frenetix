"""
This module is the main module of the Occlusion calculation of the reactive planner.
All submodules are started and managed from here

Author: Korbinian Moller, TUM
Date: 15.04.2023
"""

# imports
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
from functools import reduce

# commonroad inputs
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as ohf
from commonroad_rp.occlusion_planning.utils.visualization import OccPlot
from commonroad_rp.occlusion_planning.visibility_module import VisibilityModule
from commonroad_rp.occlusion_planning.visibility_estimator import OccVisibilityEstimator
from commonroad_rp.occlusion_planning.uncertainty_map_evaluator import OccUncertaintyMapEvaluator
from commonroad_rp.occlusion_planning.phantom_module import OccPhantomModule


class OcclusionModule:
    def __init__(self, scenario, config, ref_path, log_path, planner):
        self.scenario_id = scenario.scenario_id
        self.log_path = log_path
        self.debug_mode = config.debug.debug_mode
        self.plot = config.occlusion.show_occlusion_plot
        if config.occlusion.scope == "sensor_radius":
            self.scope = config.prediction.sensor_radius
        else:
            self.scope = config.occlusion.scope

        self.occ_scenario = OccScenario(ego_width=config.vehicle.width,
                                        ego_length=config.vehicle.length,
                                        ref_path=ref_path,
                                        scenario_lanelet_network=scenario.lanelet_network,
                                        dt=config.planning.dt,
                                        sidewalk_buffer=2)  # if sidewalk buffer > 0 sidewalks will be considered

        self.vis_module = VisibilityModule(scenario=scenario,
                                           lanelets=self.occ_scenario.lanelet_polygon,
                                           sensor_radius=config.prediction.sensor_radius)

        if config.occlusion.use_occlusion_module:

            self.occ_visible_area = OccArea(area_type="visible")
            self.occ_occluded_area = OccArea(area_type="occluded")

            self.occ_map = OccMap(debug_mode=config.debug.debug_mode)

            if self.plot:
                self.occ_plot = OccPlot(config=config, log_path=self.log_path,
                                        scenario_id=self.scenario_id, occ_scenario=self.occ_scenario)

            self.occ_phantom_module = OccPhantomModule(config=config,
                                                       occ_scenario=self.occ_scenario,
                                                       vis_module=self.vis_module,
                                                       occ_visible_area=self.occ_visible_area,
                                                       occ_plot=self.occ_plot,
                                                       params_risk=planner.params_risk,
                                                       params_harm=planner.params_harm)

            self.occ_uncertainty_map_evaluator = OccUncertaintyMapEvaluator(self.vis_module, self.occ_map,
                                                                            self.occ_plot)

            self.occ_visibility_estimator = OccVisibilityEstimator(self.occ_scenario, self.vis_module, self.occ_plot)

    def step(self):

        if self.vis_module.visible_area_timestep is not None:
            visible_area, occluded_area, add_occ_plot = \
                ohf.calc_occluded_areas(
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

        self.occ_visible_area.set_area(visible_area)
        self.occ_occluded_area.set_area(occluded_area)

        self.occ_map.step(self.occ_visible_area.points, self.occ_occluded_area.points)

        if self.plot:
            self.occ_plot.step_plot(time_step=self.vis_module.time_step,
                                    ego_pos=self.vis_module.ego_pos,
                                    ref_path=self.occ_scenario.ref_path,
                                    lanelet_polygon=self.occ_scenario.lanelets_single,
                                    sidewalk_polygon=self.occ_scenario.sidewalk_combined,
                                    lanelet_polygon_along_path=self.occ_scenario.lanelets_along_path_combined,
                                    visible_area_vm=self.vis_module.visible_area_timestep,
                                    obstacles=self.vis_module.obstacles,
                                    visible_area=self.occ_visible_area.poly,
                                    occluded_area=self.occ_occluded_area.poly,
                                    # occlusion_map=self.occ_map.map_detail,
                                    # additional_plot=add_occ_plot,
                                    error_flag=self.occ_map.error)

        return self.occ_map.map_detail


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


class OccMap:
    def __init__(self, debug_mode=0):
        self._np_init_len = 3
        self._threshold_low = -100
        self._threshold_high = 10
        self.error = False
        self.map_detail = None
        self.map = None
        self.debug_mode = debug_mode
        self.log = []

    def step(self, points_visible, points_occluded):

        self.error = False
        points_visible, points_occluded = self._check_points(points_visible, points_occluded)

        if self.map is None:
            self._initialize(points_visible, points_occluded)
        else:
            if points_visible is None:
                if self.debug_mode >= 2:
                    print("No visible area available, Occlusion map will be increased by 1")
                    self._increase_map()
            else:
                self._update(points_visible, points_occluded)

        self.log.append(self.map)

    def _check_points(self, points_visible, points_occluded):
        if points_occluded is None:
            points_occluded = np.empty([0, self._np_init_len])
        return points_visible, points_occluded

    def _increase_map(self):
        self.map_detail[:, 3] += 1
        # Limit updated map to upper threshold
        self.map_detail = ohf.np_replace_max_value_with_new_value(self.map_detail, 3, self._threshold_high,
                                                                  self._threshold_high)
        self.map_detail[:, 4] = 1 - self.map_detail[:, 3] / self._threshold_high
        self.map = self.map_detail[:, [1, 2, 3]]

    def _update(self, np_visible, np_occluded):
        # new visible area has higher priority than old area --> -100, if value is added it will
        # still be below zero --> visible
        np_visible = np.c_[
            np_visible, np.ones(np_visible.shape[0]) * self._threshold_low, np.zeros(np_visible.shape[0])]
        np_occluded = np.c_[np_occluded, np.ones(np_occluded.shape[0]), np.zeros(np_occluded.shape[0])]

        # create a new occlusion map --> it's needed to identify the area of interest
        occ_map_new = np.concatenate((np_visible, np_occluded), axis=0)

        # save hash values in variable --> needed for the coordinate comparison (np only supports comparison in 1d)
        hash_map = self.map_detail[:, 0]
        hash_map_new = occ_map_new[:, 0]

        # find hashes (representation of coordinates) and corresponding indices that exist in both arrays
        hash_both, idx_map_new, idx_map = np.intersect1d(hash_map_new, hash_map, return_indices=True)

        # get values at shared coordinates from OLD occlusion map
        common_values_old_map = self.map_detail[idx_map, :]

        # get values at shared coordinates from NEW occlusion map
        common_values_new_map = occ_map_new[idx_map_new, :]

        # add occlusion values (stored in column 3) from old map to new map
        common_values_new_map[:, 3] += common_values_old_map[:, 3]
        common_values_new_map_0 = ohf.np_replace_negative_with_zero(common_values_new_map, 3)

        # find hashes and the index of values that only exist in NEW map
        hash_only_new = np.setdiff1d(hash_map_new, hash_map)
        idx_only_occ_map_new = np.where(np.isin(occ_map_new[:, 0], hash_only_new))[0]

        # get coordinates and corresponding values that ONLY exist in NEW map (and replace negative values with 0)
        occ_map_new_only = occ_map_new[idx_only_occ_map_new, :]

        # replace occlusion values of new points with the highest possible number (occ_threshold)
        occ_map_new_only = ohf.np_replace_non_negative_with_value(occ_map_new_only, 3, self._threshold_high)

        # replace values smaller than 0 with 0 (--> visible at this time_step)
        occ_map_new_only_0 = ohf.np_replace_negative_with_zero(occ_map_new_only, 3)

        # Create updated map
        occ_map_updated = np.concatenate((common_values_new_map_0, occ_map_new_only_0), axis=0)

        # Limit updated map to upper threshold
        occ_map_updated = ohf.np_replace_max_value_with_new_value(occ_map_updated, 3,
                                                                  self._threshold_high, self._threshold_high)

        # Calculate relative occlusion value
        occ_map_updated[:, 4] = 1 - occ_map_updated[:, 3] / self._threshold_high

        self.map_detail = occ_map_updated
        self.map = self.map_detail[:, [1, 2, 3]]

    def _initialize(self, np_visible, np_occluded):
        # hash, x, y, absolute value from 0(visible) to threshold (occluded), relative value from 0(unknown) to 1
        # (visible)

        if np_visible is None:
            if self.debug_mode >= 2:
                print("No visible area available, Occlusion cannot be initialized!")
                self.error = True
                return

        np_visible = np.c_[np_visible, np.zeros(np_visible.shape[0]), np.ones(np_visible.shape[0])]
        np_occluded = np.c_[np_occluded, np.ones(np_occluded.shape[0]) * self._threshold_high,
                            np.zeros(np_occluded.shape[0])]
        # if no occlusion map exists, the initial occlusion map is ready after this step
        self.map_detail = np.concatenate((np_visible, np_occluded), axis=0)
        self.map = self.map_detail[:, [1, 2, 3]]
