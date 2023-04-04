# imports
import numpy as np
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as ohf
import commonroad_rp.occlusion_planning.utils.vis_helper_functions as vhf
from commonroad_rp.occlusion_planning.occlusion_obstacles import EstimationObstacle
from shapely.geometry import Point


class OccVisibilityEstimator:
    def __init__(self, occ_scenario=None, vis_module=None, occ_plot=None):
        self.occ_scenario = occ_scenario
        self.vis_module = vis_module
        self.occ_plot = occ_plot
        self.forecast_distance_s = None
        self.cost_factor = 100
        self.costs = None

    def evaluate_trajectories(self, trajectories, predictions, forecast_timestep=10,
                              forecast_distance_d=0.5, forecast_sample_d=6, consideration_radius=30):

        # set self.costs to None
        self.costs = None

        # calculate visibility ratio -> np.array [v_ratio, d in curvilinear coords]
        v_ratio = self.estimate(predictions,
                                forecast_timestep=forecast_timestep,
                                forecast_distance_d=forecast_distance_d,
                                forecast_sample_d=forecast_sample_d,
                                consideration_radius=consideration_radius)

        # set s in global curvilinear coordinates (for interpolation of d)
        s_interp = trajectories[0].curvilinear.s[0] + self.forecast_distance_s

        # initialize cost list
        costs = []

        # iterate over trajectories
        for traj in trajectories:
            # interpolate d for each trajectory using s_interp
            d_interp = np.interp(s_interp, traj.curvilinear.s, traj.curvilinear.d)

            # interpolate v_ratio for each trajectory
            v_ratio_interp = np.interp(d_interp, v_ratio[:, 1], v_ratio[:, 0])

            # calculate occ ratio (1-v_ratio) for cost calculation
            occ_ratio = (1 - v_ratio_interp)
            cost = self.cost_factor * occ_ratio
            costs.append(cost)

        self.costs = ohf.normalize_costs_z(costs, 50)

        self.occ_plot.plot_trajectories_cost_color(trajectories, self.costs)

    def estimate(self, predictions, forecast_timestep=10, forecast_distance_d=1.5, forecast_sample_d=2,
                 consideration_radius=None) -> np.array:

        # define consideration_radius
        if consideration_radius is None:
            consideration_radius = self.vis_module.sensor_radius

        # set forecast distance in curvilinear coordinates (s), if v=0 standard value of 5m will be used
        if self.vis_module.ego_state.velocity > 0:
            forecast_distance_s = self.vis_module.ego_state.velocity * self.occ_scenario.dt * forecast_timestep
        else:
            forecast_distance_s = 5

        # calculate (forecast_sample_d * 2) estimation points around forecast_distance_s on ref path
        estimation_points, forecast_s_np = self._create_estimation_points(forecast_distance_s, forecast_distance_d,
                                                                          forecast_sample_d)

        # create estimation obstacles based on predictions and timestep
        estimation_obstacles = _create_estimation_obstacles(predictions, forecast_timestep)

        # calculate reference_area for comparison of estimation_points (sensor radius around forecast_s_point)
        reference_area = self.vis_module.lanelets.intersection(Point(forecast_s_np).buffer(consideration_radius))

        # calculate v_ratios
        v_ratios = self._calc_v_ratios(estimation_points, estimation_obstacles, reference_area)

        # store forecast_distance in object
        self.forecast_distance_s = forecast_distance_s

        return v_ratios

    def _calc_v_ratios(self, estimation_points, estimation_obstacles, reference_area) -> np.array:
        # calculate visible area for each estimation point
        v_ratios = []
        for point in estimation_points:
            point.evaluate(reference_area, self.vis_module.sensor_radius, estimation_obstacles,
                           self.occ_scenario, ax=self.occ_plot.ax)

            self.occ_plot.ax.plot(point.pos[0], point.pos[1], 'bo')

            # store v_ratios
            if point.v_ratio is not None:
                v_ratios.append([point.v_ratio, point.d])
            else:
                v_ratios.append([point.v_ratio_basic, point.d])

        # convert v_ratios to numpy array
        v_ratios = np.array(v_ratios)

        return v_ratios

    def _create_estimation_points(self, forecast_distance_s, forecast_distance_d, forecast_sample_d):
        estimation_points = []
        ref_path_ls = self.occ_scenario.ref_path_ls
        ego_pos_point = Point(self.vis_module.ego_pos)

        # find point with delta s to ego pos on reference_path
        forecast_s_np = np.array(ref_path_ls.interpolate(ref_path_ls.project(ego_pos_point) +
                                                         forecast_distance_s).coords).flatten()

        # find normal vector on forecast_point_s
        normal_vector, _ = ohf.normal_vector_at_point(ref_path_ls, forecast_s_np)
        normal_vector = ohf.unit_vector(normal_vector)

        # find sample points next to forecast point s with distance d
        for i in range(-forecast_sample_d, forecast_sample_d + 1):
            sample_point = forecast_s_np + normal_vector * i * forecast_distance_d

            # create estimation point if sample is within the lanelet network (sidewalk shall not be considered)
            if Point(sample_point).within(self.occ_scenario.lanelets_combined):
                estimation_point = OccVisEstimationPoint(sample_point, i * forecast_distance_d)
                estimation_points.append(estimation_point)

        return estimation_points, forecast_s_np


class OccVisEstimationPoint:
    def __init__(self, pos, d):
        self.v_ratio = None
        self.v_ratio_basic = None
        self.d = d
        self.pos = pos
        self.visible_area = None
        self.occluded_area = None
        self.visible_areas_detail = dict()
        self.occluded_areas_detail = dict()

    def evaluate(self, reference_area, sensor_radius, estimation_obstacles, occ_scenario, ax):

        self._calc_areas(reference_area, sensor_radius, estimation_obstacles, occ_scenario.ref_path)

        self._evaluate_areas(prio_area=occ_scenario.lanelets_along_path_combined,
                             prio_weight=5,
                             sidewalk=occ_scenario.sidewalk_combined,
                             sidewalk_weight=1,
                             ax=None)

    def _calc_areas(self, lanelet_polygon, sensor_radius, estimation_obstacles, ref_path):

        # calculate visible area at estimation point position
        self.visible_area = vhf.calc_visible_area_from_lanelet_geometry(lanelet_polygon, self.pos, sensor_radius=None)

        # Calculate occlusions from obstacles and subtract them from visible_area --> updated visible area at position
        self.visible_area = vhf.calc_visible_area_from_obstacle_occlusions(self.visible_area, self.pos,
                                                                           estimation_obstacles, sensor_radius)

        # Update the visible area to the area of interest and calculate occluded areas
        self.visible_area, self.occluded_area, _ = ohf.calc_occluded_areas(
            ego_pos=self.pos,
            visible_area=self.visible_area,
            ref_path=ref_path,
            lanelets=lanelet_polygon,
            scope=None,  # use entire given lanelet --> reference area
            ref_path_buffer=15,
            cut_backwards=True)

    def _evaluate_areas(self, std_weight=1, prio_area=None, prio_weight=1, sidewalk=None, sidewalk_weight=1, ax=None):

        # load values to temp variables
        visible_area_m2 = self.visible_area.area
        occluded_area_m2 = self.occluded_area.area

        # calculate total area
        area_total_m2 = visible_area_m2 + occluded_area_m2
        self.v_ratio_basic = visible_area_m2 / area_total_m2

        # exit function here if both "special" areas are none --> only v_ratio_basic is calculated
        if prio_area is None and sidewalk is None:
            return

        # initialize variables
        visible_prio_area_m2 = 0
        visible_sw_area_m2 = 0
        occluded_prio_area_m2 = 0
        occluded_sw_area_m2 = 0

        if prio_area is not None:
            # evaluate visible area
            visible_prio_area = prio_area.intersection(self.visible_area)
            visible_prio_area_m2 = visible_prio_area.area
            self.visible_areas_detail["visible_prio_area"] = visible_prio_area

            # evaluate occluded area
            occluded_prio_area = prio_area.intersection(self.occluded_area)
            occluded_prio_area_m2 = occluded_prio_area.area
            self.occluded_areas_detail["occluded_prio_area"] = occluded_prio_area

        if sidewalk is not None:
            # evaluate visible area
            visible_sw_area = sidewalk.intersection(self.visible_area)
            visible_sw_area_m2 = visible_sw_area.area
            self.visible_areas_detail["visible_sw_area"] = visible_sw_area

            # evaluate occluded area
            occluded_sw_area = sidewalk.intersection(self.occluded_area)
            occluded_sw_area_m2 = occluded_sw_area.area
            self.occluded_areas_detail["occluded_sw_area"] = occluded_sw_area

        # subtract "special" areas from "standard" areas
        if prio_area is not None and sidewalk is not None:
            subtract_visible = visible_prio_area.union(visible_sw_area)
            subtract_occluded = occluded_prio_area.union(occluded_sw_area)
        elif prio_area is not None:
            subtract_visible = visible_prio_area
            subtract_occluded = occluded_prio_area
        elif sidewalk is not None:
            subtract_visible = visible_sw_area
            subtract_occluded = occluded_sw_area

        visible_standard_area = self.visible_area.difference(subtract_visible)
        visible_standard_area_m2 = visible_standard_area.area

        occluded_standard_area = self.occluded_area.difference(subtract_occluded)
        occluded_standard_area_m2 = occluded_standard_area.area

        sum_visible = (visible_prio_area_m2 * prio_weight +
                       visible_sw_area_m2 * sidewalk_weight +
                       visible_standard_area_m2 * std_weight)

        sum_occluded = (occluded_prio_area_m2 * prio_weight +
                        occluded_sw_area_m2 * sidewalk_weight +
                        occluded_standard_area_m2 * std_weight)

        self.v_ratio = sum_visible / (sum_visible + sum_occluded)

        if ax is not None:
            ax.plot(self.pos[0], self.pos[1], 'mo')
            ohf.plot_polygons(ax, prio_area, 'k')

            ohf.fill_polygons(ax, visible_prio_area, 'g')
            ohf.fill_polygons(ax, visible_standard_area, 'yellow')

            ohf.fill_polygons(ax, occluded_prio_area, 'r')
            ohf.fill_polygons(ax, occluded_standard_area, 'darkorange')

            ohf.fill_polygons(ax, visible_sw_area, 'lime')
            ohf.fill_polygons(ax, occluded_sw_area, 'coral')


def _create_estimation_obstacles(predictions, forecast_timestep) -> list:
    # prepare estimations (e.g. from WaleNet) for further calculation
    estimation_obstacles = []
    for i in predictions:
        # create estimation obstacles and save them to predictions_prepared
        est_obst = EstimationObstacle(i, predictions[i], forecast_timestep)
        estimation_obstacles.append(est_obst)
        # ohf.fill_polygons(self.occ_plot.ax, est_obst.polygon, 'orange', zorder=100)

    return estimation_obstacles

# eof
