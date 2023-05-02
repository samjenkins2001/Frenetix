"""
This module is the main module of the Phantom obstacle generation of the reactive planner.

Author: Korbinian Moller, TUM
Date: 19.04.2023
"""

# imports
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point, LineString

# commonroad imports
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as ohf
import commonroad_rp.occlusion_planning.utils.vis_helper_functions as vhf
from commonroad_rp.occlusion_planning.basic_modules.occlusion_obstacles import OccPhantomObstacle
from commonroad.scenario.obstacle import ObstacleType

# risk assessment and harm imports
from risk_assessment.harm_estimation import get_harm
from risk_assessment.helpers.timers import ExecTimer
from risk_assessment.helpers.collision_helper_function import create_tvobstacle
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectories_collision_dynamic_obstacles


class OccPhantomModule:
    def __init__(self, config=None, occ_scenario=None, vis_module=None,
                 occ_visible_area=None, occ_plot=None, params_risk=None, params_harm=None, debug_mode=0):
        self.config = config.occlusion
        self.debug_mode = debug_mode
        self.ego_vehicle_params = config.vehicle
        self.occ_scenario = occ_scenario
        self.vis_module = vis_module
        self.occ_visible_area = occ_visible_area
        self.occ_plot = occ_plot
        self.params_risk = params_risk
        self.params_harm = params_harm
        self.max_number_of_phantom_peds = 1
        self.ped_width = 0.5
        self.ped_length = 0.3
        self.ped_velocity = 1.11
        self.sorted_static_obstacles = None
        self.phantom_peds = None
        self.ego_pos_s = None
        self.max_s_trajectories = None
        self.costs = None
        self.max_costs = 10

        # variable to store commonroad like predictions
        self.cr_predictions = dict()

        # calculate the maximum distance from a vehicle corner point to the vehicle center point
        self.ego_diag = np.sqrt(
            np.power(self.occ_scenario.ego_length, 2) + np.power(self.occ_scenario.ego_width, 2))

    ################################
    # Main Function of phantom module
    ################################

    def evaluate_trajectories(self, trajectories, max_harm=0.5, plot=False):

        # calculate spawn points and create phantom obstacles
        self._evaluation_preprocessing(trajectories)

        # if no phantom obstacles exist
        if self.phantom_peds is None or len(self.phantom_peds) == 0:
            return trajectories, np.zeros(len(trajectories))

        # calc harm for each trajectory
        traj_valid, harm_valid, traj_invalid, harm_invalid = self._calc_harm(trajectories, max_harm, plot)

        # debug message
        if self.debug_mode > 0:
            print('<ReactivePlanner>: Rejected {} infeasible trajectories due to phantom pedestrian harm'
                  .format(len(traj_invalid)))

        # convert harm value to costs
        self.costs = ohf.normalize_costs_z(harm_valid, self.max_costs)

        # visualize harm probability
        if plot and self.occ_plot is not None:
            self.occ_plot.plot_trajectory_harm_color(traj_valid, harm_valid, min_harm=0, max_harm=0.8)

        # plot phantom peds and their trajectories if plot is activated
        if plot and self.occ_plot is not None:
            self.occ_plot.plot_phantom_ped_trajectory(self.phantom_peds)

        return traj_valid, self.costs

    def _calc_harm(self, trajectories, max_harm, plot):
        # inti temp variables (ok=ok, nok = not ok)
        harm_ok = []
        harm_nok = []
        traj_ok = []
        traj_nok = []

        # create commonroad like predictions for harm estimation
        self._combine_commonroad_predictions()

        # create timer
        timer = ExecTimer(timing_enabled=False)

        # iterate over trajectories and calc harm
        for traj in trajectories:

            # calculate harm using harm model
            ego_harm_traj, obst_harm_traj = get_harm(scenario=self,
                                                     traj=traj,
                                                     predictions=self.cr_predictions,
                                                     ego_vehicle_type=ObstacleType('car'),
                                                     vehicle_params=self.ego_vehicle_params,
                                                     modes=self.params_risk,
                                                     coeffs=self.params_harm,
                                                     timer=timer)

            # find maximum harm of trajectory
            traj_harm = max(max(h) for h in obst_harm_traj.values())
            if traj_harm > max_harm:
                traj_nok.append(traj)
                harm_nok.append(traj_harm)
            else:
                traj_ok.append(traj)
                harm_ok.append(traj_harm)

            # visualize collision if enabled
            if self.config.visualize_collision and self.occ_plot is not None:
                collision = self._check_trajectory_collision(traj, mode='shapely')
                self.occ_plot.plot_phantom_collision(collision)

        return traj_ok, harm_ok, traj_nok, harm_nok

    ################################
    # create phantom obstacles
    ################################

    def _evaluation_preprocessing(self, trajectories):

        # only consider static obstacles (ped won't be in front of a moving car)
        self.sorted_static_obstacles = [obst for obst in self.vis_module.obstacles if obst.obstacle_role == 'STATIC']

        # quit here, if no static obstacle exists
        if len(self.sorted_static_obstacles) == 0:
            return

        # sort obstacles by distance to ego pos
        self.sorted_static_obstacles = sorted(self.sorted_static_obstacles, key=lambda obstacle:
                                              distance.euclidean(obstacle.pos, self.vis_module.ego_pos))

        # find max s in trajectories
        self.max_s_trajectories = ohf.find_max_s_in_trajectories(trajectories)

        # check, which obstacles area along the driving direction
        self._check_obstacles()

        # find spawn points (dict with xy and cl) and points on ref path for phantom pedestrians
        spawn_points = self._find_spawn_points()

        # create phantom pedestrians
        self._create_phantom_peds(spawn_points)

        # calculate trajectories of phantom peds
        self._calc_trajectories()

    def _create_phantom_peds(self, spawn_points):
        # create empty list for phantom pedestrians
        phantom_peds = []

        # iterate over spawn points
        for i, spawn_point in enumerate(spawn_points):
            # create vector to calculate orientation
            vector = spawn_point['rp_xy'] - spawn_point['xy']

            # calculate orientation with vector and reference vector [1, 0]
            orientation = vhf.angle_between(np.array([1, 0]), vector)

            # find needed parameters for phantom ped
            create_cr_obst = True
            calc_ped_traj_polygons = False

            if self.config.visualize_collision:
                calc_ped_traj_polygons = True

            # create phantom pedestrian and add to list
            phantom_peds.append(OccPhantomObstacle(i + 1, spawn_point['xy'], orientation, self.ped_length,
                                                   self.ped_width, vector, s=spawn_point['cl'][0],
                                                   calc_ped_traj_polygons=calc_ped_traj_polygons,
                                                   create_cr_obst=create_cr_obst))

        # assign local variable to object
        self.phantom_peds = phantom_peds

    def _calc_trajectories(self):
        for ped in self.phantom_peds:
            ped.set_velocity(self.ped_velocity)

            ped.calc_goal_position(self.occ_scenario.sidewalk_combined)
            ped.calc_trajectory(dt=self.occ_scenario.dt)

    def _check_obstacles(self):

        for obst in self.sorted_static_obstacles:
            if obst.polygon.intersects(self.occ_visible_area.poly.buffer(0.1)):
                obst.visible_in_driving_direction = True
            else:
                obst.visible_in_driving_direction = False

    ################################
    # find spawn points
    ################################

    def _find_spawn_points(self, shift_spawn_points=True):

        self.ego_pos_s = self.occ_scenario.ref_path_ls.project(Point(self.vis_module.ego_pos))
        spawn_points = []

        # iterate over obstacles (sorted by distance to ego position)
        for obst in self.sorted_static_obstacles:

            # if the obstacle is visible (and not behind the vehicle), find relevant corner points
            if obst.visible_in_driving_direction:

                # find relevant corner points (cp) for each obstacle and the corresponding vectors
                # to ref path (needed later)
                relevant_cp = self._spawn_point_preprocessing(obst, offset=self.ped_width / 2 * 1.1)

                # calc curvilinear coordinates for relevant_cp and remove points that are too far away
                relevant_cp = self._convert_to_cl_and_remove_far_points(relevant_cp)

                # check if more than one possible spawn point exists for each obstacle
                if len(relevant_cp) > 1:

                    # if more than one relevant corner point exists, find critical spawn point and vector_to_ref_path
                    spawn_point = self._find_relevant_spawn_point(relevant_cp)

                    # add spawn point to spawn_points array
                    spawn_points.append(spawn_point)

                elif len(relevant_cp) == 1:
                    # only one relevant corner exists -> add relevant corner to spawn_points array
                    spawn_points.append(relevant_cp[0])

                else:
                    continue

        # limit spawn points to max_number_of_phantom_peds
        if len(spawn_points) > self.max_number_of_phantom_peds:
            # sort spawn points according to their distance to the ego position
            sorted_spawn_points_with_index = sorted(enumerate(spawn_points), key=lambda sp:
            distance.euclidean(sp[1]['xy'], self.vis_module.ego_pos))

            # convert combined list to separate lists
            sorted_indices, sorted_spawn_points = zip(*sorted_spawn_points_with_index)

            # select relevant points
            spawn_points = sorted_spawn_points[:self.max_number_of_phantom_peds]

        if shift_spawn_points and len(spawn_points) > 0:
            ref_path_ls = self.occ_scenario.ref_path_ls
            for spawn_point in spawn_points:
                # create line from spawn point to point on ref path to find intersection with visible area
                line = np.stack([spawn_point['xy'], spawn_point['rp_xy']])
                line_ls = LineString(line)

                # find shifted spawn point as shapely point
                spawn_point_shifted_point = self.vis_module.visible_area_timestep.buffer(self.ped_length/2 * 1.2). \
                    exterior.intersection(line_ls)

                # if no shifted spawn point could be found, use spawn point at corner
                if not spawn_point_shifted_point.is_empty:
                    # calc curvilinear coordinates of shifted spawn point
                    s = ref_path_ls.project(spawn_point_shifted_point)
                    d = ref_path_ls.distance(spawn_point_shifted_point)

                    # overwrite values in spawn point
                    spawn_point['xy'] = np.array(spawn_point_shifted_point.coords).flatten()
                    spawn_point['cl'] = np.array([s, d])

        return spawn_points

    def _convert_to_cl_and_remove_far_points(self, relevant_cp):

        # initialize variables
        ref_path_ls = self.occ_scenario.ref_path_ls
        relevant_cp_new = []

        # iterate through corner points and calculate curvilinear s and d coordinate for each corner
        for cp in relevant_cp:
            s = ref_path_ls.project(Point(cp['xy']))
            d = ref_path_ls.distance(Point(cp['xy']))

            # only add points that can be reached by a trajectory
            if not s >= self.max_s_trajectories and not s < self.ego_pos_s:
                relevant_cp_dict = {'xy': cp['xy'], 'cl': np.array([s, d]), 'rp_xy': cp['rp_xy']}
                relevant_cp_new.append(relevant_cp_dict)

        return relevant_cp_new

    def _find_relevant_spawn_point(self, relevant_cp) -> dict:
        """
        This function uses the list of relevant corner points and finds the most critical spawn point for a phantom
        pedestrian. Only one spawn point will be determined for each obstacle.
        Args:
            relevant_cp: list of dicts with relevant corner points {'xy' - corner points in xy coordinates,
                                                                    'cl' - corner points in curvilinear coordinates,
                                                                    'rp_xy' - point on ref path in xy coordinates}


        Returns: dict of spawn point {xy, cl, rp_xy}

        """
        # convert list of curvilinear coordinates to numpy array [s,d]
        relevant_cp_cl = np.array([d['cl'] for d in relevant_cp])

        # threshold, that points are "close" (only s considered)
        threshold = 0.2

        # calc distance of s coordinates between corner_cl and ego pos (projected on ref path)
        distances = abs(relevant_cp_cl[:, 0] - self.ego_pos_s)

        # find index of the closest point to ego pos (only s considered)
        min_distance_index = np.argmin(distances)

        # check whether there is a point, that has a similar s coordinate
        similar_indices = np.where(np.isclose(distances, distances[min_distance_index], rtol=0, atol=threshold))[0]

        # if two points have similar s coordinates, the point with the smaller d coordinate shall be used
        if len(similar_indices) > 1:
            # find point with smaller d coordinate
            min_d_index = np.argmin(relevant_cp_cl[similar_indices, 1])

            # update index
            min_distance_index = similar_indices[min_d_index]

        # assign spawn point
        spawn_point = relevant_cp[min_distance_index]

        return spawn_point

    def _spawn_point_preprocessing(self, obst, offset=0.25) -> list:
        """
        This function calculates relevant corner points and returns them as a list (for each obstacle)
        Args:
            obst: obstacle (one obstacle of type OcclusionObstacle)

        Returns: - list of dicts with relevant corner points [xy, cl, rp](potential spawn points for phantom peds)

        """
        # load ref path linestring from occ scenario
        ref_path_ls = self.occ_scenario.ref_path_ls

        # initialize list for relevant corner points
        relevant_corner_points = []

        # if offset shall be used, calculate corner points from polygon, otherwise use corner points of obstacle
        if offset > 0:
            corners = np.array(obst.polygon.buffer(offset, join_style=2).exterior.coords[:-1])
        else:
            corners = obst.corner_points

        # iterate over corner points
        for corner in corners:

            # prepare needed variables to calculate points and vectors
            cp = Point(corner)
            point_on_ref_path = np.array(ref_path_ls.interpolate(ref_path_ls.project(cp)).coords).flatten()
            vector = point_on_ref_path - corner
            normal_vector = ohf.unit_vector(vector)

            # calc points towards and in the opposite direction to ref path
            point_to_rp = corner + 0.1 * normal_vector
            point_opp_rp = corner - 0.1 * normal_vector

            # if the opposite point is within the visible area, the corner is not relevant
            if Point(point_opp_rp).within(self.vis_module.visible_area_timestep.buffer(0.01)):
                continue

            # if the opposite point is within the obstacle, the corner is not relevant
            if Point(point_opp_rp).within(obst.polygon.buffer(-0.01)):
                continue

            # if the point towards the ref path is within the obstacle, the corner is not relevant
            if Point(point_to_rp).within(obst.polygon.buffer(-0.01)):
                continue

            # if the point is outside the lanelet network (including sidewalk), the corner is not relevant
            if not Point(point_opp_rp).within(self.occ_scenario.lanelets_combined_with_sidewalk):
                continue

            # check if phantom pedestrian would be within another obstacle
            if cp.buffer(self.ped_width / 2).intersects(self.vis_module.obstacles_polygon):
                continue

            # if corner point is a potential spawn point for phantom peds, add to list
            relevant_cp_dict = {'xy': corner, 'cl': None, 'rp_xy': point_on_ref_path}
            relevant_corner_points.append(relevant_cp_dict)

        return relevant_corner_points

    ################################
    # Functions for harm model
    ################################

    def obstacle_by_id(self, obstacle_id):
        for ped in self.phantom_peds:
            if ped.obstacle_id == obstacle_id:
                return ped.commonroad_dynamic_obstacle

        return None

    def time_variant_collision_object_by_id(self, obstacle_id):
        for ped in self.phantom_peds:
            if ped.obstacle_id == obstacle_id:
                return ped.cr_tv_collision_object

        return None

    def _combine_commonroad_predictions(self):
        # clear prediction dict
        self.cr_predictions = {}

        # add ped predictions to dict
        for ped in self.phantom_peds:
            self.cr_predictions[ped.obstacle_id] = ped.commonroad_predictions

    ################################
    # helper functions
    ################################

    def _check_trajectory_collision(self, traj, mode='shapely') -> dict or None:
        """
        Checks whether a trajectory leads to a collision with a phantom pedestrian in the scenario.
        Args:
            traj: commonroad trajectory sample
            mode: mode whether to use shapely or Commonroad pycrcc

        Returns:
            dict with collision information
        """
        # check if mode is valid
        valid_modes = ['shapely', 'pycrcc', 'None']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {', '.join(valid_modes)}")

        # init ego trajectory polygons
        ego_traj_polygons = None

        # init collision dict
        collision_dict = dict()

        # iterate over phantom peds
        for ped in self.phantom_peds:

            collision_dict[ped.obstacle_id] = {'collision': False, 'collision_timestep': None, 'ped_id': None,
                                               'ped': None, 'traj': traj, 'ego_traj_polygons': None}

            # check for collision using pycrcc
            if mode == 'pycrcc':

                # create time variant collision object
                ego_tvo = create_tvobstacle(traj_list=np.array([traj.cartesian.x,
                                                                traj.cartesian.y,
                                                                traj.cartesian.theta]).transpose().tolist(),
                                            box_length=self.occ_scenario.ego_length / 2,
                                            box_width=self.occ_scenario.ego_width / 2,
                                            start_time_step=0)

                # check trajectory for collision with phantom ped (returns timestep of collision or -1)
                coll_check_cr = trajectories_collision_dynamic_obstacles([ego_tvo], [ped.cr_tv_collision_object])[0]

                # if collision is true, update collision_dict and return
                if coll_check_cr != -1:
                    collision_dict[ped.obstacle_id] = {'collision': True, 'collision_timestep': coll_check_cr,
                                                       'ped_id': ped.obstacle_id, 'ped': ped, 'traj': traj,
                                                       'ego_traj_polygons': None}

            # check for collisions using shapely
            elif mode == 'shapely':
                # check if collision is possible (s coordinates match) and perform detailed collision check
                if np.any((traj.curvilinear.s + self.ego_diag / 2) >= (ped.s - ped.diag / 2)):

                    # if trajectory has not been converted to polygon list, create polygon list
                    if ego_traj_polygons is None:
                        ego_traj_polygons = ohf.compute_vehicle_polygons(traj.cartesian.x, traj.cartesian.y,
                                                                         traj.cartesian.theta,
                                                                         self.occ_scenario.ego_width,
                                                                         self.occ_scenario.ego_length)

                    # store polygon list in local variable and shorten it
                    ped_traj_polygons = ped.traj_polygons['polygons'][:len(ego_traj_polygons['polygons'])]

                    # iterate through polygons and check for collision
                    for i, (ego_poly, ped_poly) in enumerate(zip(ego_traj_polygons['polygons'], ped_traj_polygons)):

                        if ego_poly.intersects(ped_poly):
                            # if collision is detected return collision
                            collision_dict[ped.obstacle_id] = {'collision': True, 'collision_timestep': i,
                                                               'ped_id': ped.obstacle_id, 'ped': ped, 'traj': traj,
                                                               'ego_traj_polygons': ego_traj_polygons}

                            # exit for loop
                            break

            else:
                return

        return collision_dict
